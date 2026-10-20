// banjo-nx (package 27c): the fills read before each completion point, the fix of the one-frame flashes.
//
// Package 28b removed banjo_renut_check_worker (the stream checks hashed by a worker thread at each kick): on the
// console it answered none of the render thread's checks (0 of about 300,000 every 10 s) while hashing 3.4 GB on core
// 0. The render thread hashes its checks itself, as before package 27c.
#pragma once

namespace banjo::renut_gpu::checks {

// Game threads: a kick (KickDevice, the Swap's, a callback) for the render thread: the interval's vertex and texture
// fills are read there, before the fence or callback that lets the game rewrite them completes.
void QueueKickFlush();

}  // namespace banjo::renut_gpu::checks
