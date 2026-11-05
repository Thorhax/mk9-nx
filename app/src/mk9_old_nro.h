// mk9-nx 1.0.4: the game's two names in its folder. Since 1.0.3 the game is mk9-nx/mk9-nx.nro (the Homebrew Menu shows
// a folder as an app by the NRO named after it); before, mk9.nro. An update left the old mk9.nro beside the new one,
// and forwarders made for it (and sphaira's list) kept starting the old game: testers of 1.0.3 still had the sign-in
// prompt it fixed. So the two are kept the same game.

#pragma once

namespace mk9::old_nro {

// Switch, first thing at start. Started as mk9-nx.nro: an mk9.nro beside it that is not this game is replaced by a
// copy of it (on a thread of its own, once: about 90 MB). Started as mk9.nro (a forwarder made before 1.0.3): when
// mk9-nx.nro beside it is a newer version, this file becomes a copy of that one, a dialog asks to start the game again
// and the process ends here (it does not return). Elsewhere it does nothing.
void AtStart();

}  // namespace mk9::old_nro
