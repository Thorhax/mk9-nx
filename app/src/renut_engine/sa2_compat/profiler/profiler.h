// sa2-nx (2026-10-20): banjo-nx's on-screen profiler, which Renut's frame statistics (renut_engine/perf) feed. SA2 has
// no such window: the calls do nothing.
#pragma once

namespace banjo::profiler {

inline void SetGpuFrameMs(double) {}
inline void SetRenderThreadMs(double) {}
inline void RecordHostPresent() {}
inline void SetRendererInfo(const char*, const char*) {}

}  // namespace banjo::profiler
