// Package pool implements compute pool routing for Crush.
//
// When a user passes --model "pool/qwen2.5:7b", the pool resolver
// health-checks configured Ollama nodes in priority order and
// routes to the first healthy node that has the requested model
// AND has it registered in the Crush provider config.
package pool

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"sort"
	"strings"
	"time"
)

// Node represents a single compute pool node (an Ollama instance).
type Node struct {
	Provider  string `json:"provider"`   // Must match a provider ID in crush.json (e.g., "ollama-msi")
	HealthURL string `json:"health_url"` // Ollama /api/tags endpoint (e.g., "http://localhost:11435/api/tags")
	Priority  int    `json:"priority"`   // Lower number = higher priority
}

// Config represents the compute pool configuration.
type Config struct {
	Nodes []Node `json:"nodes"`
}

// ProviderModels maps provider IDs to their registered model IDs from crush.json.
// This is used to cross-check that a model is both available on the hardware
// and registered in the Crush config.
type ProviderModels map[string][]string

// ollamaModel is a single model from Ollama's /api/tags response.
type ollamaModel struct {
	Name string `json:"name"`
}

// ollamaTagsResponse is Ollama's /api/tags JSON shape.
type ollamaTagsResponse struct {
	Models []ollamaModel `json:"models"`
}

// IsPoolModel returns true if the model string uses pool routing (starts with "pool/").
func IsPoolModel(modelStr string) bool {
	return strings.HasPrefix(modelStr, "pool/")
}

// ModelID extracts the model name from a pool model string.
// "pool/qwen2.5:7b" -> "qwen2.5:7b"
func ModelID(modelStr string) string {
	return strings.TrimPrefix(modelStr, "pool/")
}

// Resolve takes a pool model string and returns the resolved "provider/model"
// string by health-checking nodes in priority order.
//
// A node is eligible only if:
//  1. It responds to health checks (Ollama /api/tags)
//  2. The Ollama instance has the model loaded/available
//  3. The model is registered in the Crush provider config for that node
//
// Example: Resolve(cfg, registered, "pool/qwen2.5:7b") -> "ollama-msi/qwen2.5:7b"
func Resolve(poolCfg Config, registered ProviderModels, modelStr string) (string, error) {
	modelID := ModelID(modelStr)

	if len(poolCfg.Nodes) == 0 {
		return "", fmt.Errorf("pool: no nodes configured — add a \"pool\" section to crush.json")
	}

	// Sort by priority (lower = preferred).
	nodes := make([]Node, len(poolCfg.Nodes))
	copy(nodes, poolCfg.Nodes)
	sort.Slice(nodes, func(i, j int) bool {
		return nodes[i].Priority < nodes[j].Priority
	})

	client := &http.Client{Timeout: 2 * time.Second}

	for _, node := range nodes {
		// Check 1: Is the model registered in the Crush config for this provider?
		if !isRegistered(registered, node.Provider, modelID) {
			slog.Debug("Pool: model not in provider config", "provider", node.Provider, "model", modelID)
			continue
		}

		// Check 2: Is the node healthy and does it actually have the model?
		available, err := nodeHasModel(client, node, modelID)
		if err != nil {
			slog.Debug("Pool: node unreachable", "provider", node.Provider, "error", err)
			continue
		}
		if available {
			resolved := fmt.Sprintf("%s/%s", node.Provider, modelID)
			slog.Info("Pool: routed", "model", modelID, "node", node.Provider)
			return resolved, nil
		}
		slog.Debug("Pool: node healthy but missing model on hardware", "provider", node.Provider, "model", modelID)
	}

	return "", fmt.Errorf("pool: no healthy node has model %q (checked %d nodes)", modelID, len(nodes))
}

// isRegistered checks if a model ID is in the provider's registered model list.
func isRegistered(registered ProviderModels, provider, modelID string) bool {
	models, ok := registered[provider]
	if !ok {
		return false
	}
	for _, m := range models {
		if m == modelID {
			return true
		}
	}
	return false
}

// nodeHasModel checks if a node is healthy and has the requested model.
func nodeHasModel(client *http.Client, node Node, modelID string) (bool, error) {
	resp, err := client.Get(node.HealthURL)
	if err != nil {
		return false, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return false, fmt.Errorf("HTTP %d", resp.StatusCode)
	}

	var tags ollamaTagsResponse
	if err := json.NewDecoder(resp.Body).Decode(&tags); err != nil {
		return false, fmt.Errorf("decode: %w", err)
	}

	for _, m := range tags.Models {
		if m.Name == modelID {
			return true, nil
		}
	}
	return false, nil
}

// ResolveOrPassthrough resolves pool models, passes non-pool models through unchanged.
func ResolveOrPassthrough(poolCfg Config, registered ProviderModels, modelStr string) (string, error) {
	if !IsPoolModel(modelStr) {
		return modelStr, nil
	}
	return Resolve(poolCfg, registered, modelStr)
}
