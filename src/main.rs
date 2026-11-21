//! wdelta v2 — lossless compression for bf16 safetensors, with delta coding against a base model.
//!
//! Per tensor:
//!   REF    identical to the base tensor of the same name -> stored as a reference
//!   DELTA  bf16, same size as base -> d = ord(target) - ord(base), tokenised and rANS coded
//!          with one table per base exponent
//!   EXP    bf16 with no usable base -> exponent byte rANS coded, sign+mantissa stored raw
//!   RAW    anything else
//! The decoded file is byte-identical to the input; a SHA-256 is stored and checked.
//! Use base "none" to compress without a base model (EXP/RAW only).
//!
//! WDZ3 identifies the base: every REF/DELTA record carries the xxh3-128 hash of the base
//! tensor it needs, checked before that tensor is used; the file carries an optional base
//! label (env WDELTA_BASE_ID at encode time) and a fingerprint over all base tensors used.

use sha2::{Digest, Sha256};
use xxhash_rust::xxh3::{xxh3_128, Xxh3};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

#[cfg(not(target_endian = "little"))]
compile_error!("wdelta assumes a little-endian host");

const MAGIC: &[u8; 4] = b"WDZ3";
const PB: u32 = 12; // probability bits
const T: u32 = 1 << PB;
const RANS_L: u32 = 1 << 23;
const DSYM: usize = 55; // delta token alphabet
const NSTATE: usize = 4; // interleaved rANS states
const CHUNK: usize = 1 << 22; // elements per independently coded chunk

const MODE_RAW: u8 = 0;
const MODE_REF: u8 = 1;
const MODE_DELTA: u8 = 2;
const MODE_EXP: u8 = 3;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

static PROF: std::sync::Mutex<[f64; 4]> = std::sync::Mutex::new([0.0; 4]); // read, sha, codec, write
fn prof(i: usize, t: Instant) { PROF.lock().unwrap()[i] += t.elapsed().as_secs_f64(); }
fn prof_print() {
    if std::env::var("WDELTA_PROF").is_ok() {
        let p = PROF.lock().unwrap();
        eprintln!("  prof: read {:.1}s  sha {:.1}s  codec {:.1}s  write {:.1}s", p[0], p[1], p[2], p[3]);
    }
}

// ------------------------------------------------------------------ byte views

fn as_bytes(v: &[u16]) -> &[u8] { unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 2) } }
fn as_bytes_mut(v: &mut [u16]) -> &mut [u8] { unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, v.len() * 2) } }

/// Tensor data, kept as u16 when the byte length is even so bf16 code needs no copy.
enum Buf { W(Vec<u16>), B(Vec<u8>) }
impl Buf {
    fn bytes(&self) -> &[u8] { match self { Buf::W(v) => as_bytes(v), Buf::B(v) => v } }
    fn words(&self) -> Option<&[u16]> { match self { Buf::W(v) => Some(v), Buf::B(_) => None } }
}

fn read_range(path: &Path, off: u64, len: u64) -> Res<Buf> {
    let mut f = File::open(path)?;
    f.seek(SeekFrom::Start(off))?;
    if len % 2 == 0 {
        let mut v = vec![0u16; (len / 2) as usize];
        f.read_exact(as_bytes_mut(&mut v))?;
        Ok(Buf::W(v))
    } else {
        let mut v = vec![0u8; len as usize];
        f.read_exact(&mut v)?;
        Ok(Buf::B(v))
    }
}

// ------------------------------------------------------------------ safetensors

struct TInfo { name: String, dtype: String, start: u64, end: u64 }
struct StFile { header_raw: Vec<u8>, data_base: u64, tensors: Vec<TInfo> }

fn read_st(path: &Path) -> Res<StFile> {
    let mut f = File::open(path)?;
    let mut nb = [0u8; 8];
    f.read_exact(&mut nb)?;
    let n = u64::from_le_bytes(nb);
    let mut js = vec![0u8; n as usize];
    f.read_exact(&mut js)?;
    let v: serde_json::Value = serde_json::from_slice(&js)?;
    let mut tensors = Vec::new();
    for (k, meta) in v.as_object().ok_or("bad header")? {
        if k == "__metadata__" { continue; }
        let off = meta["data_offsets"].as_array().ok_or("no offsets")?;
        tensors.push(TInfo {
            name: k.clone(),
            dtype: meta["dtype"].as_str().unwrap_or("").to_string(),
            start: off[0].as_u64().ok_or("off")?,
            end: off[1].as_u64().ok_or("off")?,
        });
    }
    tensors.sort_by_key(|t| t.start);
    let mut header_raw = nb.to_vec();
    header_raw.extend_from_slice(&js);
    Ok(StFile { header_raw, data_base: 8 + n, tensors })
}

struct BaseIndex { map: HashMap<String, (PathBuf, u64, u64, String)> }

fn load_base(dir: &str) -> Res<BaseIndex> {
    let mut map = HashMap::new();
    if dir == "none" { return Ok(BaseIndex { map }); }
    let dir = Path::new(dir);
    let mut files: Vec<PathBuf> = if dir.is_dir() {
        std::fs::read_dir(dir)?.filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().map(|x| x == "safetensors").unwrap_or(false)).collect()
    } else { vec![dir.to_path_buf()] };
    files.sort();
    for p in files {
        let st = read_st(&p)?;
        for t in &st.tensors {
            map.insert(t.name.clone(), (p.clone(), st.data_base + t.start, t.end - t.start, t.dtype.clone()));
        }
    }
    Ok(BaseIndex { map })
}

fn base_tensor(base: &BaseIndex, name: &str) -> Res<Option<(Buf, String)>> {
    match base.map.get(name) {
        Some((p, off, len, dt)) => Ok(Some((read_range(p, *off, *len)?, dt.clone()))),
        None => Ok(None),
    }
}

// ------------------------------------------------------------------ bf16 helpers

#[inline(always)]
fn ord(u: u16) -> i32 { if u & 0x8000 != 0 { 0x7FFF - (u & 0x7FFF) as i32 } else { u as i32 + 0x8000 } }
#[inline(always)]
fn unord(o: i32) -> u16 { if o >= 0x8000 { (o - 0x8000) as u16 } else { 0x8000 | (0x7FFF - o) as u16 } }
#[inline(always)]
fn tokenize(d: i32) -> (u8, u32, u32) {
    let a = d.unsigned_abs();
    if a <= 15 { return ((d + 15) as u8, 0, 0); }
    let k = 32 - a.leading_zeros(); // 5..=16
    ((31 + (k - 5) * 2 + (d < 0) as u32) as u8, a - (1 << (k - 1)), k - 1)
}
#[inline(always)]
fn exp_of(u: u16) -> u8 { (u >> 7) as u8 }

/// Inverse of `tokenize`: rebuild d from its token and the extra bits that follow it.
#[inline(always)]
fn detokenize(sym: u8, bits: &mut BitR) -> i32 {
    let s = sym as i32;
    if s < 31 { return s - 15; }
    let k = 5 + ((s - 31) >> 1) as u32;
    let mag = (1i32 << (k - 1)) + bits.get(k - 1) as i32;
    if (s - 31) & 1 == 1 { -mag } else { mag }
}

// ------------------------------------------------------------------ bit IO

struct BitW { out: Vec<u8>, acc: u64, n: u32 }
impl BitW {
    fn new() -> Self { BitW { out: Vec::new(), acc: 0, n: 0 } }
    #[inline(always)]
    fn put(&mut self, v: u32, bits: u32) {
        self.acc |= (v as u64) << self.n;
        self.n += bits;
        if self.n >= 32 { self.out.extend_from_slice(&(self.acc as u32).to_le_bytes()); self.acc >>= 32; self.n -= 32; }
    }
    fn finish(mut self) -> Vec<u8> {
        while self.n > 0 { self.out.push(self.acc as u8); self.acc >>= 8; self.n = self.n.saturating_sub(8); }
        self.out.extend_from_slice(&[0u8; 8]); // padding for the reader's 4-byte refills
        self.out
    }
}
struct BitR<'a> { d: &'a [u8], pos: usize, acc: u64, n: u32 }
impl<'a> BitR<'a> {
    fn new(d: &'a [u8]) -> Self { BitR { d, pos: 0, acc: 0, n: 0 } }
    #[inline(always)]
    fn get(&mut self, bits: u32) -> u32 {
        if self.n < bits {
            let w = match self.d.get(self.pos..self.pos + 4) { Some(s) => u32::from_le_bytes(s.try_into().unwrap()), None => 0 };
            self.acc |= (w as u64) << self.n;
            self.n += 32;
            self.pos += 4;
        }
        let v = (self.acc & ((1u64 << bits) - 1)) as u32;
        self.acc >>= bits; self.n -= bits;
        v
    }
}

// ------------------------------------------------------------------ serialization helpers

fn put_u16(v: &mut Vec<u8>, x: u16) { v.extend_from_slice(&x.to_le_bytes()); }
fn put_u32(v: &mut Vec<u8>, x: u32) { v.extend_from_slice(&x.to_le_bytes()); }
fn put_u64(v: &mut Vec<u8>, x: u64) { v.extend_from_slice(&x.to_le_bytes()); }
fn get_u16(d: &[u8], p: &mut usize) -> Res<u16> { let s = d.get(*p..*p + 2).ok_or("truncated")?; *p += 2; Ok(u16::from_le_bytes(s.try_into()?)) }
fn get_u32(d: &[u8], p: &mut usize) -> Res<u32> { let s = d.get(*p..*p + 4).ok_or("truncated")?; *p += 4; Ok(u32::from_le_bytes(s.try_into()?)) }
fn get_u64(d: &[u8], p: &mut usize) -> Res<u64> { let s = d.get(*p..*p + 8).ok_or("truncated")?; *p += 8; Ok(u64::from_le_bytes(s.try_into()?)) }
fn get_blob<'a>(d: &'a [u8], p: &mut usize) -> Res<&'a [u8]> { let n = get_u32(d, p)? as usize; let s = d.get(*p..*p + n).ok_or("truncated")?; *p += n; Ok(s) }

// ------------------------------------------------------------------ context-modeled rANS token coder

fn normalize(counts: &[u64]) -> Vec<u32> {
    let total: u64 = counts.iter().sum();
    let mut f = vec![0u32; counts.len()];
    if total == 0 { return f; }
    let (mut sum, mut best) = (0i64, 0usize);
    for s in 0..counts.len() {
        if counts[s] > 0 {
            f[s] = ((counts[s] as u128 * T as u128 / total as u128) as u32).max(1);
            sum += f[s] as i64;
            if counts[s] > counts[best] { best = s; }
        }
    }
    let mut diff = T as i64 - sum;
    if f[best] as i64 + diff >= 1 {
        f[best] = (f[best] as i64 + diff) as u32;
    } else {
        diff += f[best] as i64 - 1;
        f[best] = 1;
        let mut order: Vec<usize> = (0..f.len()).filter(|&s| f[s] > 1).collect();
        order.sort_by_key(|&s| std::cmp::Reverse(f[s]));
        for s in order { while diff < 0 && f[s] > 1 { f[s] -= 1; diff += 1; } }
    }
    f
}

/// Code `toks` (each < nsym) with one static table per context byte `ctx[i]`.
/// NSTATE interleaved rANS states (element i uses state i % NSTATE) share one byte stream.
fn encode_tokens(toks: &[u8], ctx: &[u8], nsym: usize) -> Vec<u8> {
    let n = toks.len();
    let mut counts = vec![0u64; 256 * nsym];
    for i in 0..n { counts[ctx[i] as usize * nsym + toks[i] as usize] += 1; }
    let mut freq = vec![0u32; 256 * nsym];
    let mut cum = vec![0u32; 256 * nsym];
    let mut out = Vec::new();
    let used: Vec<usize> = (0..256).filter(|&c| counts[c * nsym..(c + 1) * nsym].iter().any(|&x| x > 0)).collect();
    put_u16(&mut out, used.len() as u16);
    for &c in &used {
        let f = normalize(&counts[c * nsym..(c + 1) * nsym]);
        let mut acc = 0;
        let nz: Vec<usize> = (0..nsym).filter(|&s| f[s] > 0).collect();
        out.push(c as u8);
        put_u16(&mut out, nz.len() as u16);
        for s in 0..nsym { freq[c * nsym + s] = f[s]; cum[c * nsym + s] = acc; acc += f[s]; }
        for s in nz { out.push(s as u8); put_u16(&mut out, f[s] as u16); }
    }
    let mut r: Vec<u8> = Vec::with_capacity(n / 2 + 16);
    let mut x = [RANS_L; NSTATE];
    for i in (0..n).rev() {
        let j = i % NSTATE;
        let k = ctx[i] as usize * nsym + toks[i] as usize;
        let f = freq[k];
        let x_max = ((RANS_L >> PB) << 8) * f;
        while x[j] >= x_max { r.push(x[j] as u8); x[j] >>= 8; }
        x[j] = ((x[j] / f) << PB) + (x[j] % f) + cum[k];
    }
    for j in (0..NSTATE).rev() { for k in 0..4 { r.push((x[j] >> (8 * k)) as u8); } }
    r.reverse();
    r.extend_from_slice(&[0u8; 4]); // padding: decoder may read a little past the end on corrupt input
    put_u32(&mut out, r.len() as u32);
    out.extend_from_slice(&r);
    out
}

#[derive(Clone, Copy, Default)]
struct Slot { freq: u16, bias: u16, sym: u8 }

/// Decode n tokens; `ctx(i)` gives element i's context, `sink(i, sym)` receives each token.
/// Generic so the per-element work is inlined into the rANS loop (one pass, no buffers).
#[inline(always)]
fn decode_tokens_with<C: Fn(usize) -> u8, S: FnMut(usize, u8)>(data: &[u8], p: &mut usize, n: usize, nsym: usize, ctx: C, mut sink: S) -> Res<()> {
    let nctx = get_u16(data, p)? as usize;
    if nctx > 256 { return Err("bad table count".into()); }
    let mut ctx_of = [u16::MAX; 256];
    let mut tbl = vec![Slot::default(); nctx * T as usize];
    for c in 0..nctx {
        let id = *data.get(*p).ok_or("truncated")? as usize; *p += 1;
        let nz = get_u16(data, p)? as usize;
        ctx_of[id] = c as u16;
        let mut acc = 0u32;
        for _ in 0..nz {
            let s = *data.get(*p).ok_or("truncated")?; *p += 1;
            let f = get_u16(data, p)? as u32;
            if acc + f > T || (s as usize) >= nsym { return Err("bad table".into()); }
            for k in 0..f { tbl[c * T as usize + (acc + k) as usize] = Slot { freq: f as u16, bias: k as u16, sym: s }; }
            acc += f;
        }
        if acc != T { return Err("bad table".into()); }
    }
    // contexts never seen by the encoder point at an empty sentinel; reject them up front per element
    let r = get_blob(data, p)?;
    if r.len() < 4 * NSTATE + 4 { return Err("short stream".into()); }
    let mut x0 = u32::from_be_bytes(r[0..4].try_into()?);
    let mut x1 = u32::from_be_bytes(r[4..8].try_into()?);
    let mut x2 = u32::from_be_bytes(r[8..12].try_into()?);
    let mut x3 = u32::from_be_bytes(r[12..16].try_into()?);
    let mut rp = 16usize;
    let rl = r.len() - 4;
    macro_rules! step {
        ($x:ident, $i:expr) => {{
            let c = ctx_of[ctx($i) as usize];
            if c == u16::MAX { return Err("unknown context".into()); }
            let e = tbl[c as usize * T as usize + ($x & (T - 1)) as usize];
            sink($i, e.sym);
            $x = e.freq as u32 * ($x >> PB) + e.bias as u32;
            while $x < RANS_L {
                if rp >= rl { return Err("stream overrun".into()); }
                $x = ($x << 8) | r[rp] as u32; rp += 1;
            }
        }};
    }
    let mut i = 0;
    while i + 3 < n { step!(x0, i); step!(x1, i + 1); step!(x2, i + 2); step!(x3, i + 3); i += 4; }
    if i < n { step!(x0, i); i += 1; }
    if i < n { step!(x1, i); i += 1; }
    if i < n { step!(x2, i); }
    Ok(())
}

// ------------------------------------------------------------------ chunk codecs

fn encode_delta_chunk(a: &[u16], b: &[u16]) -> Vec<u8> {
    let n = a.len();
    let mut toks = vec![0u8; n];
    let mut ctx = vec![0u8; n];
    let mut bits = BitW::new();
    for i in 0..n {
        let (t, ev, eb) = tokenize(ord(a[i]) - ord(b[i]));
        toks[i] = t; ctx[i] = exp_of(b[i]);
        bits.put(ev, eb);
    }
    let mut out = Vec::new();
    put_u32(&mut out, n as u32);
    out.extend_from_slice(&encode_tokens(&toks, &ctx, DSYM));
    let bb = bits.finish();
    put_u32(&mut out, bb.len() as u32);
    out.extend_from_slice(&bb);
    out
}

fn decode_delta_chunk(data: &[u8], b: &[u16], a: &mut [u16]) -> Res<()> {
    let mut p = 0;
    let n = get_u32(data, &mut p)? as usize;
    if n != b.len() || n != a.len() { return Err("chunk size mismatch".into()); }
    // locate the bit stream (it follows the token stream) before decoding tokens
    let mut q = p;
    let nctx = get_u16(data, &mut q)? as usize;
    for _ in 0..nctx { q += 1; let nz = get_u16(data, &mut q)? as usize; q += 3 * nz; }
    let _ = get_blob(data, &mut q)?;
    let mut bits = BitR::new(get_blob(data, &mut q)?);
    decode_tokens_with(data, &mut p, n, DSYM, |i| exp_of(b[i]), |i, sym| {
        a[i] = unord(ord(b[i]) + detokenize(sym, &mut bits));
    })
}

fn encode_exp_chunk(a: &[u16]) -> Vec<u8> {
    let n = a.len();
    let toks: Vec<u8> = a.iter().map(|&u| exp_of(u)).collect();
    let sm: Vec<u8> = a.iter().map(|&u| (((u >> 8) & 0x80) | (u & 0x7F)) as u8).collect();
    let mut out = Vec::new();
    put_u32(&mut out, n as u32);
    out.extend_from_slice(&encode_tokens(&toks, &vec![0u8; n], 256));
    put_u32(&mut out, n as u32);
    out.extend_from_slice(&sm);
    out
}

fn decode_exp_chunk(data: &[u8], a: &mut [u16]) -> Res<()> {
    let mut p = 0;
    let n = get_u32(data, &mut p)? as usize;
    if n != a.len() { return Err("chunk size mismatch".into()); }
    let mut q = p;
    let nctx = get_u16(data, &mut q)? as usize;
    for _ in 0..nctx { q += 1; let nz = get_u16(data, &mut q)? as usize; q += 3 * nz; }
    let _ = get_blob(data, &mut q)?;
    let sm = get_blob(data, &mut q)?;
    if sm.len() != n { return Err("bad sm stream".into()); }
    decode_tokens_with(data, &mut p, n, 256, |_| 0, |i, t| {
        let s = sm[i] as u16;
        a[i] = ((s & 0x80) << 8) | ((t as u16) << 7) | (s & 0x7F);
    })
}

// ------------------------------------------------------------------ parallel chunking

fn par_map<F: Fn(usize) -> R + Sync, R: Send>(n: usize, f: F) -> Vec<R> {
    let threads = std::thread::available_parallelism().map(|x| x.get()).unwrap_or(2).min(n.max(1));
    let next = AtomicUsize::new(0);
    let mut results: Vec<Option<R>> = (0..n).map(|_| None).collect();
    let slots: Vec<std::sync::Mutex<&mut Option<R>>> = results.iter_mut().map(std::sync::Mutex::new).collect();
    std::thread::scope(|sc| {
        for _ in 0..threads {
            sc.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= n { break; }
                let r = f(i);
                **slots[i].lock().unwrap() = Some(r);
            });
        }
    });
    drop(slots);
    results.into_iter().map(|r| r.unwrap()).collect()
}

fn pack_chunks(chunks: Vec<Vec<u8>>) -> Vec<u8> {
    let mut out = Vec::new();
    put_u32(&mut out, chunks.len() as u32);
    for c in &chunks { put_u64(&mut out, c.len() as u64); }
    for c in chunks { out.extend_from_slice(&c); }
    out
}

/// Decode `nel` u16s from a chunked payload; `f(chunk_bytes, lo, out_slice)` decodes one chunk.
fn unpack_chunks<F: Fn(&[u8], usize, &mut [u16]) -> Res<()> + Sync>(data: &[u8], nel: usize, f: F) -> Res<Vec<u16>> {
    let mut p = 0;
    let nch = get_u32(data, &mut p)? as usize;
    if nch != nel.div_ceil(CHUNK) { return Err("chunk count mismatch".into()); }
    let mut spans = Vec::with_capacity(nch);
    let lens: Vec<usize> = (0..nch).map(|_| get_u64(data, &mut p).map(|x| x as usize)).collect::<Res<_>>()?;
    for l in lens { let e = p.checked_add(l).ok_or("bad chunk length")?; let s = data.get(p..e).ok_or("truncated")?; spans.push(s); p = e; }
    let mut a = vec![0u16; nel];
    {
        let parts: Vec<std::sync::Mutex<&mut [u16]>> = a.chunks_mut(CHUNK).map(std::sync::Mutex::new).collect();
        let errs = par_map(nch, |i| { let mut g = parts[i].lock().unwrap(); f(spans[i], i * CHUNK, &mut g).map_err(|e| e.to_string()) });
        for e in errs { e?; }
    }
    Ok(a)
}

fn encode_delta(a: &[u16], b: &[u16]) -> Vec<u8> {
    pack_chunks(par_map(a.len().div_ceil(CHUNK), |i| {
        let lo = i * CHUNK; let hi = (lo + CHUNK).min(a.len());
        encode_delta_chunk(&a[lo..hi], &b[lo..hi])
    }))
}
fn encode_exp(a: &[u16]) -> Vec<u8> {
    pack_chunks(par_map(a.len().div_ceil(CHUNK), |i| {
        let lo = i * CHUNK; let hi = (lo + CHUNK).min(a.len());
        encode_exp_chunk(&a[lo..hi])
    }))
}

// ------------------------------------------------------------------ file level

#[derive(Default)]
struct Stats { n: [usize; 4], bytes_in: [u64; 4], bytes_out: [u64; 4] }

fn encode_file(base: &BaseIndex, input: &Path, output: &Path) -> Res<()> {
    let t0 = Instant::now();
    let st = read_st(input)?;
    let flen = std::fs::metadata(input)?.len();
    let data_len = flen - st.data_base;
    let mut sha = Sha256::new();
    sha.update(&st.header_raw);
    let mut w = BufWriter::new(File::create(output)?);
    w.write_all(MAGIC)?;
    let base_id = std::env::var("WDELTA_BASE_ID").unwrap_or_default();
    w.write_all(&(base_id.len() as u16).to_le_bytes())?;
    w.write_all(base_id.as_bytes())?;
    let mut fp = Xxh3::new();
    w.write_all(&(st.header_raw.len() as u64).to_le_bytes())?;
    w.write_all(&st.header_raw)?;
    w.write_all(&data_len.to_le_bytes())?;
    let mut recs: Vec<(u64, u64, Option<&TInfo>)> = Vec::new();
    let mut pos = 0;
    for t in &st.tensors {
        if t.start > pos { recs.push((pos, t.start, None)); }
        recs.push((t.start, t.end, Some(t)));
        pos = t.end;
    }
    if pos < data_len { recs.push((pos, data_len, None)); }
    w.write_all(&(recs.len() as u32).to_le_bytes())?;
    let mut stats = Stats::default();
    for (s, e, t) in recs {
        let tgt = read_range(input, st.data_base + s, e - s)?;
        sha.update(tgt.bytes());
        let (mut mode, mut payload, mut name) = (MODE_RAW, Vec::new(), String::new());
        let mut bhash: u128 = 0;
        if let Some(t) = t {
            let is_bf16 = t.dtype == "BF16" && tgt.words().is_some();
            if let Some((bt, bdt)) = base_tensor(base, &t.name)? {
                bhash = xxh3_128(bt.bytes());
                if bt.bytes() == tgt.bytes() {
                    mode = MODE_REF; name = t.name.clone();
                } else if is_bf16 && bdt == "BF16" && bt.bytes().len() == tgt.bytes().len() {
                    let p = encode_delta(tgt.words().unwrap(), bt.words().unwrap());
                    if p.len() < tgt.bytes().len() { mode = MODE_DELTA; payload = p; name = t.name.clone(); }
                }
            }
            if mode == MODE_RAW && is_bf16 && tgt.bytes().len() >= 4096 {
                let p = encode_exp(tgt.words().unwrap());
                if p.len() < tgt.bytes().len() { mode = MODE_EXP; payload = p; }
            }
        }
        if mode == MODE_RAW { payload = tgt.bytes().to_vec(); }
        w.write_all(&[mode])?;
        w.write_all(&(e - s).to_le_bytes())?;
        w.write_all(&(name.len() as u16).to_le_bytes())?;
        w.write_all(name.as_bytes())?;
        if !name.is_empty() {
            w.write_all(&bhash.to_le_bytes())?;
            fp.update(name.as_bytes()); fp.update(&bhash.to_le_bytes());
        }
        w.write_all(&(payload.len() as u64).to_le_bytes())?;
        w.write_all(&payload)?;
        let m = mode as usize;
        stats.n[m] += 1; stats.bytes_in[m] += e - s; stats.bytes_out[m] += payload.len() as u64;
    }
    let fingerprint = fp.digest128();
    w.write_all(&fingerprint.to_le_bytes())?;
    w.write_all(&sha.finalize())?;
    w.flush()?;
    drop(w);
    let olen = std::fs::metadata(output)?.len();
    let secs = t0.elapsed().as_secs_f64();
    println!("{} -> {}", input.display(), output.display());
    println!("  {:.3} GB -> {:.3} GB = {:.2}% of original  (encode {:.0} MB/s)",
             flen as f64 / 1e9, olen as f64 / 1e9, 100.0 * olen as f64 / flen as f64, flen as f64 / 1e6 / secs);
    if stats.n[1] + stats.n[2] > 0 {
        println!("  base: {} tensors used, fingerprint {:032x}{}", stats.n[1] + stats.n[2], fingerprint,
                 if base_id.is_empty() { String::new() } else { format!(", label \"{base_id}\"") });
    }
    let pct = |m: usize| if stats.bytes_in[m] > 0 { 100.0 * stats.bytes_out[m] as f64 / stats.bytes_in[m] as f64 } else { 0.0 };
    println!("  tensors: {} delta ({:.2}%), {} exp-coded ({:.2}%), {} identical, {} raw ({:.3} GB)",
             stats.n[2], pct(2), stats.n[3], pct(3), stats.n[1], stats.n[0], stats.bytes_in[0] as f64 / 1e9);
    Ok(())
}

fn decode_file(base: &BaseIndex, input: &Path, output: Option<&Path>) -> Res<()> {
    let t0 = Instant::now();
    let mut f = std::io::BufReader::with_capacity(1 << 20, File::open(input)?);
    let flen = std::fs::metadata(input)?.len();
    let mut pos: u64 = 0;
    let mut rd = |n: usize| -> Res<Vec<u8>> {
        if n as u64 > flen.saturating_sub(pos) { return Err("file is truncated or damaged".into()); }
        let mut v = vec![0u8; n]; f.read_exact(&mut v)?; pos += n as u64; Ok(v)
    };
    if rd(4)? != MAGIC { return Err("not a WDZ3 file".into()); }
    let u64_of = |v: Vec<u8>| -> u64 { u64::from_le_bytes(v.try_into().unwrap()) };
    let bl = u16::from_le_bytes(rd(2)?.try_into().unwrap()) as usize;
    let base_id = String::from_utf8(rd(bl)?)?;
    let want_base = if base_id.is_empty() { "the base model it was made with".to_string() } else { format!("base model \"{base_id}\"") };
    let mut fp = Xxh3::new();
    let hl = u64_of(rd(8)?);
    let header = rd(hl as usize)?;
    let _data_len = u64_of(rd(8)?);
    let nrec = u32::from_le_bytes(rd(4)?.try_into().unwrap());
    // hashing and writing run on a second thread, overlapped with decoding the next tensor
    let (tx, rxc) = std::sync::mpsc::sync_channel::<Buf>(1);
    // create the output before the writer thread starts, so cleanup after an error always finds it
    let out_file = match output { Some(p) => Some(File::create(p)?), None => None };
    let header_c = header.clone();
    let consumer = std::thread::spawn(move || -> Result<[u8; 32], String> {
        let mut sha = Sha256::new();
        sha.update(&header_c);
        let mut w = out_file.map(|f| BufWriter::with_capacity(1 << 22, f));
        if let Some(w) = w.as_mut() { w.write_all(&header_c).map_err(|e| e.to_string())?; }
        for buf in rxc {
            let t = Instant::now(); sha.update(buf.bytes()); prof(1, t);
            let t = Instant::now(); if let Some(w) = w.as_mut() { w.write_all(buf.bytes()).map_err(|e| e.to_string())?; } prof(3, t);
        }
        if let Some(mut w) = w { w.flush().map_err(|e| e.to_string())?; }
        Ok(sha.finalize().into())
    });
    let mut total = header.len() as u64;
    for _ in 0..nrec {
        let mode = rd(1)?[0];
        let nbytes = u64_of(rd(8)?);
        let nl = u16::from_le_bytes(rd(2)?.try_into().unwrap()) as usize;
        let name = String::from_utf8(rd(nl)?)?;
        let bhash: u128 = if nl > 0 { u128::from_le_bytes(rd(16)?.try_into().unwrap()) } else { 0 };
        if nl > 0 { fp.update(name.as_bytes()); fp.update(&bhash.to_le_bytes()); }
        let pl = u64_of(rd(8)?);
        let payload = rd(pl as usize)?;
        let t = Instant::now();
        // fetch and check the base tensor this record depends on
        let base_buf: Option<Buf> = if nl > 0 {
            let (bt, _) = base_tensor(base, &name)?
                .ok_or_else(|| format!("tensor \"{name}\" is missing from the supplied base; this file needs {want_base}"))?;
            let got = xxh3_128(bt.bytes());
            if got != bhash {
                return Err(format!("base mismatch at tensor \"{name}\": this file needs {want_base} \
                                    (expected hash {bhash:032x}, supplied base has {got:032x})").into());
            }
            Some(bt)
        } else { None };
        let buf: Buf = match mode {
            MODE_RAW => Buf::B(payload),
            MODE_REF => { prof(0, t); base_buf.ok_or("missing base")? }
            MODE_DELTA => {
                let bt = base_buf.ok_or("missing base")?;
                prof(0, t);
                let t = Instant::now();
                let b = bt.words().ok_or("base not bf16")?;
                if (b.len() as u64) * 2 != nbytes { return Err(format!("damaged record for {name}").into()); }
                let a = unpack_chunks(&payload, b.len(), |d, lo, out| decode_delta_chunk(d, &b[lo..lo + out.len()], out))?;
                prof(2, t);
                Buf::W(a)
            }
            MODE_EXP => {
                if nbytes % 2 != 0 || (nbytes / 2) as usize > payload.len() { return Err(format!("damaged record for {name}").into()); }
                let a = unpack_chunks(&payload, (nbytes / 2) as usize, |d, _, out| decode_exp_chunk(d, out))?;
                prof(2, t);
                Buf::W(a)
            }
            _ => return Err("bad mode".into()),
        };
        if buf.bytes().len() as u64 != nbytes { return Err(format!("size mismatch in {name}").into()); }
        total += nbytes;
        tx.send(buf).map_err(|_| "writer thread failed")?;
    }
    drop(tx);
    let got = consumer.join().map_err(|_| "writer thread panicked")??;
    let fp_file = u128::from_le_bytes(rd(16)?.try_into().unwrap());
    if fp_file != fp.digest128() { return Err("base fingerprint mismatch (file damaged?)".into()); }
    let want = rd(32)?;
    let ok = got.as_slice() == want.as_slice();
    let secs = t0.elapsed().as_secs_f64();
    println!("{}: decoded {:.3} GB in {:.1}s ({:.0} MB/s), SHA-256 {}",
             input.display(), total as f64 / 1e9, secs, total as f64 / 1e6 / secs, if ok { "MATCH" } else { "MISMATCH" });
    prof_print();
    if !ok { return Err("checksum mismatch".into()); }
    Ok(())
}

/// Reimplementation of ZipLLM's BitX (bitx_bytes.rs, bitx_compress) for head-to-head
/// size comparison: XOR with base, split into an exponent byte stream and a
/// sign+mantissa byte stream, zstd on each. Identical tensors cost 0 (tensor dedup);
/// tensors without a bf16 base match get zstd on the raw bytes.
/// Streams in blocks into one zstd frame per stream, so output size equals the bulk
/// version while memory stays bounded.
struct Count(u64);
impl Write for Count {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> { self.0 += b.len() as u64; Ok(b.len()) }
    fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
}

fn read_bytes(path: &Path, off: u64, len: u64) -> Res<Vec<u8>> {
    let mut f = File::open(path)?;
    f.seek(SeekFrom::Start(off))?;
    let mut v = vec![0u8; len as usize];
    f.read_exact(&mut v)?;
    Ok(v)
}

fn bitx_size(base: &BaseIndex, input: &Path, level: i32) -> Res<()> {
    const BLK: u64 = 64 << 20; // bytes per block (even)
    let t0 = Instant::now();
    let st = read_st(input)?;
    let flen = std::fs::metadata(input)?.len();
    let (mut tout, mut nb, mut ni, mut nz) = (st.header_raw.len() as u64, 0, 0, 0);
    for t in &st.tensors {
        let len = t.end - t.start;
        let toff = st.data_base + t.start;
        let bmeta = base.map.get(&t.name).filter(|(_, _, bl, _)| *bl == len);
        match bmeta {
            Some((bp, boff, _, bdt)) if bdt == "BF16" && t.dtype == "BF16" && len % 2 == 0 => {
                let mut ee = zstd::stream::Encoder::new(Count(0), level)?;
                let mut es = zstd::stream::Encoder::new(Count(0), level)?;
                let mut identical = true;
                let mut pos = 0;
                while pos < len {
                    let n = BLK.min(len - pos);
                    let a = read_bytes(input, toff + pos, n)?;
                    let b = read_bytes(bp, boff + pos, n)?;
                    if a != b { identical = false; }
                    let m = (n / 2) as usize;
                    let mut ex = vec![0u8; m];
                    let mut sm = vec![0u8; m];
                    for i in 0..m {
                        let x = u16::from_le_bytes([a[2 * i], a[2 * i + 1]]) ^ u16::from_le_bytes([b[2 * i], b[2 * i + 1]]);
                        ex[i] = ((x >> 7) & 0xFF) as u8;
                        sm[i] = ((((x >> 15) & 1) << 7) as u8) | (x & 0x7F) as u8;
                    }
                    ee.write_all(&ex)?; es.write_all(&sm)?;
                    pos += n;
                }
                let (ce, cs) = (ee.finish()?.0, es.finish()?.0);
                if identical { ni += 1; } else { tout += ce + cs; nb += 1; }
            }
            Some((bp, boff, _, _)) if read_bytes(bp, *boff, len)? == read_bytes(input, toff, len)? => { ni += 1; }
            _ => {
                let mut e = zstd::stream::Encoder::new(Count(0), level)?;
                let mut pos = 0;
                while pos < len { let n = BLK.min(len - pos); e.write_all(&read_bytes(input, toff + pos, n)?)?; pos += n; }
                tout += e.finish()?.0; nz += 1;
            }
        }
    }
    println!("BITX(zstd-{level}) {}: {:.3} GB -> {:.3} GB = {:.2}% of original ({} xor, {} identical, {} zstd) {:.0}s",
             input.display(), flen as f64 / 1e9, tout as f64 / 1e9, 100.0 * tout as f64 / flen as f64,
             nb, ni, nz, t0.elapsed().as_secs_f64());
    Ok(())
}

/// Print what a .wdz file contains and which base it needs, without decoding it.
fn info_file(input: &Path) -> Res<()> {
    let mut f = std::io::BufReader::new(File::open(input)?);
    fn rdf(f: &mut impl Read, n: usize) -> Res<Vec<u8>> { let mut v = vec![0u8; n]; f.read_exact(&mut v)?; Ok(v) }
    if rdf(&mut f, 4)? != MAGIC { return Err("not a WDZ3 file".into()); }
    let bl = u16::from_le_bytes(rdf(&mut f, 2)?.try_into().unwrap()) as usize;
    let base_id = String::from_utf8(rdf(&mut f, bl)?)?;
    let hl = u64::from_le_bytes(rdf(&mut f, 8)?.try_into().unwrap());
    if hl > std::fs::metadata(input)?.len() { return Err("file is truncated or damaged".into()); }
    f.seek_relative(hl as i64)?;
    let data_len = u64::from_le_bytes(rdf(&mut f, 8)?.try_into().unwrap());
    let nrec = u32::from_le_bytes(rdf(&mut f, 4)?.try_into().unwrap());
    let (mut n, mut inb, mut outb) = ([0usize; 4], [0u64; 4], [0u64; 4]);
    for _ in 0..nrec {
        let mode = rdf(&mut f, 1)?[0] as usize;
        let nb = u64::from_le_bytes(rdf(&mut f, 8)?.try_into().unwrap());
        let nl = u16::from_le_bytes(rdf(&mut f, 2)?.try_into().unwrap()) as usize;
        rdf(&mut f, nl)?;
        if nl > 0 { rdf(&mut f, 16)?; }
        let pl = u64::from_le_bytes(rdf(&mut f, 8)?.try_into().unwrap());
        f.seek_relative(pl as i64)?;
        if mode < 4 { n[mode] += 1; inb[mode] += nb; outb[mode] += pl; }
    }
    let fp = u128::from_le_bytes(rdf(&mut f, 16)?.try_into().unwrap());
    let total_out = std::fs::metadata(input)?.len();
    println!("{}: WDZ3, {:.3} GB of tensor data stored in {:.3} GB", input.display(), data_len as f64 / 1e9, total_out as f64 / 1e9);
    if n[1] + n[2] > 0 {
        println!("  needs base: {}", if base_id.is_empty() { "(no label)" } else { &base_id });
        println!("  base fingerprint: {fp:032x} ({} base tensors)", n[1] + n[2]);
    } else {
        println!("  needs no base model");
    }
    for (m, label) in [(2, "delta"), (3, "exp-coded"), (1, "identical to base"), (0, "raw")] {
        if n[m] > 0 { println!("  {:>5} tensors {:<18} {:.3} GB -> {:.3} GB", n[m], label, inb[m] as f64 / 1e9, outb[m] as f64 / 1e9); }
    }
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let usage = "usage:\n  wdelta encode <base_dir|none> <target.safetensors> <out.wdz>\n  wdelta decode <base_dir|none> <in.wdz> [out.safetensors]   (omit output to verify only)\n  wdelta bitx   <base_dir> <target.safetensors> [zstd_level]\n  wdelta info   <in.wdz>\n(set WDELTA_BASE_ID when encoding to label the base, e.g. a Hugging Face repo id)";
    let r = match args.get(1).map(|s| s.as_str()) {
        Some("encode") if args.len() == 5 => load_base(&args[2]).and_then(|b| encode_file(&b, Path::new(&args[3]), Path::new(&args[4]))),
        Some("decode") if args.len() == 4 || args.len() == 5 => {
            let r = load_base(&args[2]).and_then(|b| decode_file(&b, Path::new(&args[3]), args.get(4).map(Path::new)));
            if r.is_err() { if let Some(o) = args.get(4) { let _ = std::fs::remove_file(o); } }
            r
        }
        Some("info") if args.len() == 3 => info_file(Path::new(&args[2])),
        Some("bitx") if args.len() >= 4 => {
            let level = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(3);
            load_base(&args[2]).and_then(|b| bitx_size(&b, Path::new(&args[3]), level))
        }
        _ => { eprintln!("{usage}"); std::process::exit(2); }
    };
    if let Err(e) = r { eprintln!("error: {e}"); std::process::exit(1); }
}

// ------------------------------------------------------------------ tests

#[cfg(test)]
mod tests {
    use super::*;

    /// Small deterministic PRNG (xorshift64*), so tests need no extra crates.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 { self.0 ^= self.0 >> 12; self.0 ^= self.0 << 25; self.0 ^= self.0 >> 27; self.0.wrapping_mul(0x2545F4914F6CDD1D) }
        fn u16(&mut self) -> u16 { (self.next() >> 32) as u16 }
        /// A plausible weight: small normal-ish bf16 value.
        fn weight(&mut self) -> u16 {
            let x = ((self.next() >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 0.08;
            ((x as f32).to_bits() >> 16) as u16
        }
    }

    const SPECIALS: [u16; 12] = [0x0000, 0x8000, 0x0001, 0x8001, 0x007F, 0x0080, 0x7F7F, 0xFF7F, 0x7F80, 0xFF80, 0x7FC0, 0xFFFF];

    fn roundtrip_delta(a: &[u16], b: &[u16]) {
        let enc = encode_delta_chunk(a, b);
        let mut out = vec![0u16; a.len()];
        decode_delta_chunk(&enc, b, &mut out).expect("decode failed");
        assert!(out == a, "delta round trip mismatch (n = {})", a.len());
    }

    #[test]
    fn ord_is_a_bijection_and_follows_float_order() {
        let mut seen = vec![false; 65536];
        for u in 0..=u16::MAX {
            let o = ord(u);
            assert!((0..65536).contains(&o));
            assert!(!seen[o as usize], "ord not injective at {u:#06x}");
            seen[o as usize] = true;
            assert_eq!(unord(o), u, "unord(ord(u)) != u at {u:#06x}");
        }
        // every non-NaN value: sorting by float value (with -0 < +0) must sort by ord too
        let mut vals: Vec<u16> = (0..=u16::MAX).filter(|&u| !f32::from_bits((u as u32) << 16).is_nan()).collect();
        vals.sort_by(|&x, &y| f32::from_bits((x as u32) << 16).total_cmp(&f32::from_bits((y as u32) << 16)));
        for w in vals.windows(2) { assert!(ord(w[0]) < ord(w[1]), "order broken between {:#06x} and {:#06x}", w[0], w[1]); }
    }

    #[test]
    fn tokenizer_round_trips_every_possible_delta() {
        let mut bits = BitW::new();
        let mut toks = Vec::new();
        for d in -65535..=65535 {
            let (t, ev, eb) = tokenize(d);
            assert!((t as usize) < DSYM, "token out of range for d = {d}");
            toks.push(t);
            bits.put(ev, eb);
        }
        let bytes = bits.finish();
        let mut r = BitR::new(&bytes);
        for (i, d) in (-65535..=65535).enumerate() { assert_eq!(detokenize(toks[i], &mut r), d); }
    }

    #[test]
    fn delta_round_trips_every_bf16_value_as_base_and_as_target() {
        let all: Vec<u16> = (0..=u16::MAX).collect();
        let mut rng = Rng(1);
        let rnd: Vec<u16> = (0..65536).map(|_| rng.u16()).collect();
        let near: Vec<u16> = all.iter().map(|&u| unord((ord(u) + (rng.u16() % 7) as i32 - 3).clamp(0, 65535))).collect();
        let rev: Vec<u16> = all.iter().rev().copied().collect();
        roundtrip_delta(&rnd, &all);   // every value as base
        roundtrip_delta(&all, &rnd);   // every value as target
        roundtrip_delta(&near, &all);  // small deltas everywhere, incl. across exponents and signs
        roundtrip_delta(&rev, &all);   // largest possible deltas
        roundtrip_delta(&all, &all);   // all-zero deltas
    }

    #[test]
    fn delta_round_trips_all_pairs_of_special_values() {
        let (mut a, mut b) = (Vec::new(), Vec::new());
        for &x in &SPECIALS { for &y in &SPECIALS { a.push(x); b.push(y); } }
        roundtrip_delta(&a, &b);
    }

    #[test]
    fn delta_round_trips_short_and_realistic_chunks() {
        let mut rng = Rng(7);
        for n in [0usize, 1, 2, 3, 4, 5, 7, 8, 9, 4097] {
            let b: Vec<u16> = (0..n).map(|_| rng.weight()).collect();
            let a: Vec<u16> = b.iter().map(|&u| if rng.next() % 3 == 0 { u } else { unord((ord(u) + (rng.u16() % 9) as i32 - 4).clamp(0, 65535)) }).collect();
            roundtrip_delta(&a, &b);
        }
    }

    #[test]
    fn exponent_mode_round_trips_every_bf16_value() {
        let all: Vec<u16> = (0..=u16::MAX).collect();
        for n in [0usize, 1, 3, 65536] {
            let a = &all[..n];
            let enc = encode_exp_chunk(a);
            let mut out = vec![0u16; n];
            decode_exp_chunk(&enc, &mut out).unwrap();
            assert!(out == a);
        }
    }

    #[test]
    fn multi_chunk_tensors_round_trip() {
        let mut rng = Rng(11);
        let n = CHUNK * 2 + 3;
        let b: Vec<u16> = (0..n).map(|_| rng.weight()).collect();
        let a: Vec<u16> = b.iter().map(|&u| unord((ord(u) + (rng.u16() % 5) as i32 - 2).clamp(0, 65535))).collect();
        let enc = encode_delta(&a, &b);
        let out = unpack_chunks(&enc, n, |d, lo, o| decode_delta_chunk(d, &b[lo..lo + o.len()], o)).unwrap();
        assert!(out == a);
        let enc = encode_exp(&a);
        let out = unpack_chunks(&enc, n, |d, _, o| decode_exp_chunk(d, o)).unwrap();
        assert!(out == a);
    }

    // ---- whole-file tests

    fn write_st(path: &Path, tensors: &[(&str, &str, Vec<usize>, Vec<u8>)], meta: bool) {
        let mut hdr = serde_json::Map::new();
        if meta { hdr.insert("__metadata__".into(), serde_json::json!({"format": "pt"})); }
        let mut off = 0usize;
        let mut data = Vec::new();
        for (name, dt, shape, bytes) in tensors {
            hdr.insert(name.to_string(), serde_json::json!({"dtype": dt, "shape": shape, "data_offsets": [off, off + bytes.len()]}));
            off += bytes.len();
            data.extend_from_slice(bytes);
        }
        let mut js = serde_json::to_vec(&hdr).unwrap();
        while (8 + js.len()) % 8 != 0 { js.push(b' '); } // safetensors pads its header with spaces
        let mut f = File::create(path).unwrap();
        f.write_all(&(js.len() as u64).to_le_bytes()).unwrap();
        f.write_all(&js).unwrap();
        f.write_all(&data).unwrap();
    }

    fn bf16_bytes(v: &[u16]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

    fn temp_dir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("wdelta-test-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// A base and a fine-tune exercising every record mode.
    fn make_pair(dir: &Path) -> (PathBuf, PathBuf) {
        let mut rng = Rng(3);
        let w: Vec<u16> = (0..300_000).map(|_| rng.weight()).collect();
        let w2: Vec<u16> = w.iter().map(|&u| unord((ord(u) + (rng.u16() % 5) as i32 - 2).clamp(0, 65535))).collect();
        let same: Vec<u16> = (0..5000).map(|_| rng.weight()).collect();
        let newt: Vec<u16> = (0..50_000).map(|_| rng.weight()).collect();
        let f32b: Vec<u8> = (0..1000).flat_map(|i| (i as f32 * 0.5).to_le_bytes()).collect();
        let f32f: Vec<u8> = (0..1000).flat_map(|i| (i as f32 * 0.25).to_le_bytes()).collect();
        let i64t: Vec<u8> = (0..100i64).flat_map(|i| i.to_le_bytes()).collect();
        let odd: Vec<u8> = vec![1, 2, 3, 4, 5];
        let base = dir.join("base.safetensors");
        let ft = dir.join("ft.safetensors");
        write_st(&base, &[("w", "BF16", vec![600, 500], bf16_bytes(&w)), ("same", "BF16", vec![5000], bf16_bytes(&same)),
                          ("f32", "F32", vec![1000], f32b), ("specials", "BF16", vec![144], bf16_bytes(&SPECIALS.repeat(12)))], true);
        let sp_ft: Vec<u16> = SPECIALS.iter().rev().copied().collect::<Vec<_>>().repeat(12);
        write_st(&ft, &[("w", "BF16", vec![600, 500], bf16_bytes(&w2)), ("same", "BF16", vec![5000], bf16_bytes(&same)),
                        ("f32", "F32", vec![1000], f32f), ("specials", "BF16", vec![144], bf16_bytes(&sp_ft)),
                        ("new", "BF16", vec![50_000], bf16_bytes(&newt)), ("ids", "I64", vec![100], i64t), ("odd", "U8", vec![5], odd)], true);
        (base, ft)
    }

    #[test]
    fn whole_file_round_trip_is_byte_identical() {
        let d = temp_dir("file");
        let (base, ft) = make_pair(&d);
        let b = load_base(base.to_str().unwrap()).unwrap();
        encode_file(&b, &ft, &d.join("ft.wdz")).unwrap();
        decode_file(&b, &d.join("ft.wdz"), Some(&d.join("out.safetensors"))).unwrap();
        assert!(std::fs::read(&ft).unwrap() == std::fs::read(d.join("out.safetensors")).unwrap());
        // without a base model
        let none = load_base("none").unwrap();
        encode_file(&none, &ft, &d.join("nb.wdz")).unwrap();
        decode_file(&none, &d.join("nb.wdz"), Some(&d.join("nb.safetensors"))).unwrap();
        assert!(std::fs::read(&ft).unwrap() == std::fs::read(d.join("nb.safetensors")).unwrap());
        // the base itself encoded against itself is all references
        encode_file(&b, &base, &d.join("self.wdz")).unwrap();
        decode_file(&b, &d.join("self.wdz"), None).unwrap();
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn decoding_against_the_wrong_base_fails() {
        let d = temp_dir("wrongbase");
        let (base, ft) = make_pair(&d);
        let b = load_base(base.to_str().unwrap()).unwrap();
        encode_file(&b, &ft, &d.join("ft.wdz")).unwrap();
        let wrong = load_base(ft.to_str().unwrap()).unwrap();
        let err = decode_file(&wrong, &d.join("ft.wdz"), None).unwrap_err().to_string();
        assert!(err.contains("base mismatch"), "unexpected error: {err}");
        let none = load_base("none").unwrap();
        let err = decode_file(&none, &d.join("ft.wdz"), None).unwrap_err().to_string();
        assert!(err.contains("missing from the supplied base"), "unexpected error: {err}");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn damaged_files_fail_cleanly() {
        let d = temp_dir("damage");
        let (base, ft) = make_pair(&d);
        let b = load_base(base.to_str().unwrap()).unwrap();
        encode_file(&b, &ft, &d.join("ft.wdz")).unwrap();
        let good = std::fs::read(d.join("ft.wdz")).unwrap();
        let bad = d.join("bad.wdz");
        let mut rng = Rng(5);
        // truncations
        for cut in (0..good.len()).step_by((good.len() / 300).max(1)) {
            std::fs::write(&bad, &good[..cut]).unwrap();
            assert!(decode_file(&b, &bad, None).is_err(), "truncated file at {cut} decoded");
        }
        // bit flips: must either fail or (flip in padding) still restore the exact original
        for _ in 0..600 {
            let mut v = good.clone();
            let i = (rng.next() as usize) % v.len();
            v[i] ^= 1 << (rng.next() % 8);
            std::fs::write(&bad, &v).unwrap();
            if decode_file(&b, &bad, Some(&d.join("bad.out"))).is_ok() {
                assert!(std::fs::read(d.join("bad.out")).unwrap() == std::fs::read(&ft).unwrap());
            }
        }
        let _ = std::fs::remove_dir_all(&d);
    }
}
