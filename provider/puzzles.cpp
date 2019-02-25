#ifndef local_puzzles_hpp
#define local_puzzles_hpp

#include "decompose.hpp"
#include "integral.hpp"
#include "ising.hpp"
#include "rank.hpp"

// TODO: include your engine headers
#include "decompose_tbb.hpp"
#include "integral_tbb.hpp"
#include "ising_tbb.hpp"
#include "rank_tbb.hpp"
#include "integral_opencl.hpp"

#include "decompose_opencl.hpp"

void puzzler::PuzzleRegistrar::UserRegisterPuzzles() {
  Register("decompose.ref", std::make_shared<puzzler::DecomposePuzzle>());
  Register("integral.ref", std::make_shared<puzzler::IntegralPuzzle>());
  Register("ising.ref", std::make_shared<puzzler::IsingPuzzle>());
  Register("rank.ref", std::make_shared<puzzler::RankPuzzle>());

  // TODO: Register more engines!
  Register("ising.tbb", std::make_shared<puzzler::IsingTbbProvider>());
  Register("integral.tbb", std::make_shared<puzzler::IntegralTbbProvider>());
  Register("decompose.tbb", std::make_shared<puzzler::DecomposeTbbProvider>());
  Register("rank.tbb", std::make_shared<puzzler::RankTbbProvider>());
  
  Register("decompose.opencl", std::make_shared<puzzler::DecomposeOpenCLProvider>());
  Register("integral.opencl", std::make_shared<puzzler::IntegralOpenCLProvider>());

  // Note that you can register the same engine twice under different names, for
  // example you could register the same engine for "ising.tbb" and "ising.opt"
}

#endif
