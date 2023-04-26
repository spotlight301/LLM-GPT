#ifndef ADDITIONS_HPP
#define ADDITIONS_HPP
#include "gptj.hpp"


// Returns the size in bytes of the state (rng, logits, embedding and kv_cache)
size_t gptj_get_state_size(gptj_model&);

// Copies the state to the specified destination address.
// Destination needs to have allocated enough memory.
// Returns the number of bytes copied
size_t gptj_copy_state_data(gptj_model&, uint8_t *dest);

// Set the state reading from the specified address
// Returns the number of bytes read
size_t gptj_set_state_data(gptj_model&, const uint8_t *src);
#endif // ADDITIONS_HPP
