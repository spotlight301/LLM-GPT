#include "ai.hpp"

#include <iostream>



int main() {
    Ai ai;
    std::cout << "Completing \"she replied that\"..." << std::endl;
    std::cout << "Using model " << ai.model_name << "..." << std::endl;
    std::cout << "> she replied that" << ai.complete("she replied that", '\n') << std::endl;
}
