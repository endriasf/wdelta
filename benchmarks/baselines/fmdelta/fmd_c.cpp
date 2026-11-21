// Minimal C entry points around FM-Delta's published C++ library (package/src), mirroring
// what its fmd.pyx compress_param/decompress_param do for float16 (type 2) arrays.
#include <cstdint>
#include <cstddef>
#include "fmd.h"
extern "C" size_t fmdc_compress(const int16_t* base, const int16_t* ft, int nx, int ny, void* out, size_t cap) {
  FMD* f = fmd_write_to_buffer(out, cap);
  f->type = FMD_TYPE_HALF; f->nx = nx; f->ny = ny; f->nz = 1; f->nf = 1;
  if (!fmd_write_header(f)) { fmd_write_close(f); return 0; }
  size_t n = fmd_write(f, base, ft);
  fmd_write_close(f);
  return n;
}
extern "C" size_t fmdc_decompress(const void* in, const int16_t* base, int16_t* out) {
  FMD* f = fmd_read_from_buffer(in);
  if (!fmd_read_header(f)) { fmd_read_close(f); return 0; }
  size_t n = fmd_read(f, out, const_cast<int16_t*>(base));
  fmd_read_close(f);
  return n;
}
