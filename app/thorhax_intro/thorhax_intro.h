// Thorhax intro: the short video every Thorhax port shows at startup (Switch only; elsewhere it does nothing).
// Self-contained so it can be copied into another port: this folder plus one call (see README.md).

#pragma once

namespace thorhax::intro {

// Starts the intro on its own threads and returns at once, so the game boots while it plays. It draws on the default
// window with a CPU framebuffer and plays its sound through audout, holding the SDK's output (RexSwitchHoldOutput) until
// it ends: meanwhile the game runs unseen, unheard and with untouched controllers. Call it on the main thread before
// anything opens the window or the audio device (at the start of ReXApp::OnPreSetup, before the graphics system is
// created). The first kUnskippableSeconds cannot be skipped; after that any button ends it. - and + together end it at
// any time.
void Start();

}  // namespace thorhax::intro
