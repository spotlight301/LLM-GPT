#include "justlm_pool.hpp"

#include <filesystem>
#include <fstream>
#include <stdexcept>



LM_SCHEDULABLE(bool) LM::InferencePool::store_slot(Slot &slot) {
    auto inference = slot.get_inference().lock();
    // Open output file
    std::ofstream f(get_slot_filename(slot.get_id()), std::ios::binary);
    // Write weights path
    auto weights_path = slot.get_weights_path();
    uint32_t weights_path_len = weights_path.size();
    f.write(reinterpret_cast<const char*>(&weights_path_len), sizeof(weights_path_len));
    f.write(weights_path.data(), weights_path.size());
    // Write params
    if (!f.write(reinterpret_cast<const char*>(&inference->params), sizeof(inference->params))) {
        LM_CORETURN false;
    }
    // Serialize instance
    try {
        inference->serialize(f);
    } catch (...) {
        LM_CORETURN false;
    }
    // Return success
    LM_CORETURN true;
}

LM_SCHEDULABLE(LM::InferencePool::Slot*) LM::InferencePool::load_slot(size_t id, Slot *suggested_slot) {
    // Open input file
    std::ifstream f(get_slot_filename(id), std::ios::binary);
    if (!f) {
        // Does not exist
        LM_CORETURN nullptr;
    }
    // Read weights path
    std::string weights_path;
    uint32_t weights_path_len;
    if (!f.read(reinterpret_cast<char*>(&weights_path_len), sizeof(weights_path_len))) {
        LM_CORETURN nullptr;
    }
    weights_path.resize(weights_path_len);
    if (!f.read(weights_path.data(), weights_path.size())) {
        LM_CORETURN nullptr;
    }
    // Read params
    LM::Inference::Params p;
    if (!f.read(reinterpret_cast<char*>(&p), sizeof(p))) {
        LM_CORETURN nullptr;
    }
    // Create instance
    auto& slot = suggested_slot?*suggested_slot:*(LM_COAWAIT get_free_slot());
    auto inference = slot.create_inference(id, weights_path, p).lock();
    // Deserialize instance
    try {
        LM_COAWAIT inference->deserialize(f);
    } catch (...) {
        slot.reset();
        LM_CORETURN nullptr;
    }
    // Return final slot
    LM_CORETURN &slot;
}

LM_SCHEDULABLE(LM::InferencePool::Slot*) LM::InferencePool::get_free_slot() {
    // Attempt to find free slot while finding oldest one
    Slot *oldest = nullptr;
    for (auto& slot : slots) {
        // Take free slot
        if (slot.is_free()) {
            LM_CORETURN &slot;
        }
        // Update oldest
        if (oldest == nullptr || slot.get_last_access() < oldest->get_last_access()) {
            oldest = &slot;
        }
    }
    // Free up oldest slot and take that one
    // Note: Since there has to be at least 1 slot, oldest is never going to be a nullptr
    LM_COAWAIT store_and_reset_slot(*oldest);
    LM_CORETURN oldest;
}

LM_SCHEDULABLE(LM::InferencePool::Slot*) LM::InferencePool::find_slot_by_id(size_t id, bool deserialize) {
    // Attempt to find given slot while finding oldest one
    Slot *oldest = nullptr;
    for (auto& slot : slots) {
        // Take slot with ID
        if (slot.get_id() == id) {
            LM_CORETURN &slot;
        }
        // Update oldest
        if (oldest == nullptr || slot.get_last_access() < oldest->get_last_access()) {
            oldest = &slot;
        }
    }
    // Slot not found, attempt to load it
    if (deserialize) {
        if (!oldest->is_free()) LM_COAWAIT store_slot(*oldest);
        if (!LM_COAWAIT load_slot(id, oldest)) {
            // In case slot loading failed, still reset slot for later use
            //TODO: Make this configurable
            oldest->reset();
        } else {
            LM_CORETURN oldest;
        }
    }
    // Slot not found
    LM_CORETURN nullptr;
}

LM_SCHEDULABLE(std::weak_ptr<LM::Inference>) LM::InferencePool::get_inference(size_t id) {
    auto slot = LM_COAWAIT find_slot_by_id(id);
    if (slot) {
        LM_CORETURN slot->get_inference(true);
    }
    LM_CORETURN {};
}

LM_SCHEDULABLE(std::weak_ptr<LM::Inference>) LM::InferencePool::get_or_create_inference(size_t id, const std::string &weights_path, const Inference::Params &p) {
    auto slot = LM_COAWAIT find_slot_by_id(id);
    if (slot) {
        LM_CORETURN slot->get_inference(true);
    }
    slot = LM_COAWAIT get_free_slot();
    LM_CORETURN slot->create_inference(id, weights_path, p);
}

LM_SCHEDULABLE(void) LM::InferencePool::delete_inference(size_t id) {
    auto slot = LM_COAWAIT find_slot_by_id(id, false);
    // Reset slot
    if (slot) {
        slot->reset();
    }
    // Delete file
    std::error_code ec;
    std::filesystem::remove(get_slot_filename(id), ec);
}

LM_SCHEDULABLE(void) LM::InferencePool::store_all() {
    for (auto& slot : slots) {
        if (slot.is_free()) continue;
        LM_COAWAIT store_slot(slot);
    }
    LM_CORETURN;
}

std::vector<size_t> LM::InferencePool::get_active_slot_ids() const {
    std::vector<size_t> fres;
    for (const auto& slot : slots) {
        fres.push_back(slot.get_id());
    }
    return fres;
}

void LM::InferencePool::cleanup() {
    // Collect files
    const auto prefix = get_slot_filename_prefix();
    for (auto& file : std::filesystem::directory_iterator(".")) {
        if (file.path().filename().string().find(prefix) != 0) continue;
        std::error_code ec;
        std::filesystem::remove(file, ec);
    }
}

template<typename TP>
std::time_t to_time_t(TP tp) {
    using namespace std::chrono;
    auto sctp = time_point_cast<system_clock::duration>(tp - TP::clock::now()
              + system_clock::now());
    return system_clock::to_time_t(sctp);
}

void LM::InferencePool::cleanup(time_t max_age) {
    const auto current_time = to_time_t(std::chrono::system_clock::now());
    // Collect files
    const auto prefix = get_slot_filename_prefix();
    for (auto& file : std::filesystem::directory_iterator(".")) {
        if (file.path().filename().string().find(prefix) != 0) continue;
        // Delete files older than max age
        if (current_time - to_time_t(file.last_write_time()) > max_age) {
            std::error_code ec;
            std::filesystem::remove(file, ec);
        }
    }
}
