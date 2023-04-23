#ifndef _JUSTLM_POOL_HPP
#define _JUSTLM_POOL_HPP
#include "justlm.hpp"

#include <string>
#include <string_view>
#include <vector>
#include <chrono>
#include <memory>
#include <optional>
#include <stdexcept>
#include <fstream>
#include <filesystem>



namespace LM {
class InferencePool {
    class Slot {
        std::unique_ptr<Inference> inference;
        size_t id;
        std::chrono::system_clock::time_point last_access;
        std::string weights_path;

    public:

        Slot() {
            reset();
        }

        void reset() {
            inference = nullptr;
            id = 0;
        }
        bool is_free() const {
            return inference == nullptr;
        }
        Inference& create_inference(size_t id, const std::string& weights_path, const Inference::Params& p) {
            this->id = id;
            this->weights_path = weights_path;
            inference = std::make_unique<Inference>(weights_path, p);
            return get_inference();
        }
        Inference& get_inference() {
            last_access = std::chrono::system_clock::now();
            return *inference.get();
        }

        auto get_id() const {
            return id;
        }
        auto get_last_access() const {
            return last_access;
        }
        std::string_view get_weights_path() const {
            return weights_path;
        }
    };
    std::vector<Slot> slots;

    std::string pool_name;
    bool store_on_destruct = false;

    std::string get_slot_filename_prefix() const {
        return "LMInferencePool_"+pool_name+'_';
    }
    std::string get_slot_filename(size_t id) const {
        return get_slot_filename_prefix()+std::to_string(id);
    }

    // Returns false on error
    bool store_slot(Slot& slot) {
        auto& inference = slot.get_inference();
        // Open output file
        std::ofstream f(get_slot_filename(slot.get_id()), std::ios::binary);
        // Write weights path
        auto weights_path = slot.get_weights_path();
        uint32_t weights_path_len = weights_path.size();
        f.write(reinterpret_cast<const char*>(&weights_path_len), sizeof(weights_path_len));
        f.write(weights_path.data(), weights_path.size());
        // Write params
        if (!f.write(reinterpret_cast<const char*>(&inference.params), sizeof(inference.params))) {
            return false;
        }
        // Serialize instance
        try {
            inference.serialize(f);
        } catch (...) {
            return false;
        }
        // Return success
        return true;
    }
    // Returns nullptr on error
    Slot *load_slot(size_t id, Slot *suggested_slot = nullptr) {
        // Open input file
        std::ifstream f(get_slot_filename(id), std::ios::binary);
        if (!f) {
            // Does not exist
            return nullptr;
        }
        // Read weights path
        std::string weights_path;
        uint32_t weights_path_len;
        if (!f.read(reinterpret_cast<char*>(weights_path_len), sizeof(weights_path_len))) {
            return nullptr;
        }
        weights_path.resize(weights_path_len);
        if (!f.read(weights_path.data(), weights_path.size())) {
            return nullptr;
        }
        // Read params
        LM::Inference::Params p;
        if (!f.read(reinterpret_cast<char*>(&p), sizeof(p))) {
            return nullptr;
        }
        // Create instance
        auto& slot = suggested_slot?*suggested_slot:get_free_slot();
        auto& inference = slot.create_inference(id, weights_path, p);
        // Deserialize instance
        try {
            inference.deserialize(f);
        } catch (...) {
            slot.reset();
            return nullptr;
        }
        // Return final slot
        return &slot;
    }

    void store_and_reset_slot(Slot& slot) {
        store_slot(slot); //TODO: Should handle errors somehow
        slot.reset();
    }

    Slot& get_free_slot() {
        // Attempt to find free slot while finding oldest one
        Slot *oldest = nullptr;
        for (auto& slot : slots) {
            // Take free slot
            if (slot.is_free()) {
                return slot;
            }
            // Update oldest
            if (oldest == nullptr || slot.get_last_access() < oldest->get_last_access()) {
                oldest = &slot;
            }
        }
        // Free up oldest slot and take that one
        // Note: Since there has to be at least 1 slot, oldest is never going to be a nullptr
        store_and_reset_slot(*oldest);
        return *oldest;
    }

    Slot* find_slot_by_id(size_t id, bool deserialize = true) {
        // Attempt to find given slot while finding oldest one
        Slot *oldest = nullptr;
        for (auto& slot : slots) {
            // Take free slot
            if (slot.get_id() == id) {
                return &slot;
            }
            // Update oldest
            if (oldest == nullptr || slot.get_last_access() < oldest->get_last_access()) {
                oldest = &slot;
            }
        }
        // Slot not found, attempt to load it
        if (deserialize) {
            return load_slot(id, oldest);
        }
        // Slot not found
        return nullptr;
    }

public:
    // The pool_name must be unique amonst all applications in cwd
    InferencePool(size_t size, const std::string& pool_name, bool clean_up = true) : pool_name(pool_name) {
        // Make sure size isn't zero
        if (size == 0) size = 1;
        // Create slots as requested
        slots.reserve(size);
        for (size_t c = 0; c != size; c++) {
            slots.emplace_back(Slot());
        }
        // Clean up previous slots as requested
        if (clean_up) {
            const auto prefix = get_slot_filename_prefix();
            for (const auto& file : std::filesystem::directory_iterator(".")) {
                if (file.path().filename().string().find(prefix) == 0) {
                    std::error_code ec;
                    std::filesystem::remove(file, ec);
                }
            }
        }
    }
    ~InferencePool() {
        if (store_on_destruct) {
            store_all();
        }
    }

    Inference &create_inference(size_t id, const std::string& weights_path, const Inference::Params& p) {
        auto& slot = get_free_slot();
        return slot.create_inference(id, weights_path, p);
    }
    std::optional<std::reference_wrapper<Inference>> get_inference(size_t id) {
        auto slot = find_slot_by_id(id);
        if (slot) {
            return slot->get_inference();
        }
        return {};
    }
    void delete_inference(size_t id) {
        auto slot = find_slot_by_id(id, false);
        // Reset slot
        if (slot) {
            slot->reset();
        }
        // Delete file
        std::error_code ec;
        std::filesystem::remove(get_slot_filename(id), ec);
    }
    void store_all() {
        for (auto& slot : slots) {
            store_slot(slot);
        }
    }

    void set_store_on_destruct(bool v) {
        store_on_destruct = v;
    }
    bool is_stored_on_destruction() const {
        return store_on_destruct;
    }
};
}
#endif // _JUSTLM_POOL_HPP
