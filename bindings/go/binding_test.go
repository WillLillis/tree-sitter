package tree_sitter_tree_sitter_test

import (
	"testing"

	tree_sitter "github.com/smacker/go-tree-sitter"
	"github.com/tree-sitter/tree-sitter-tree_sitter"
)

func TestCanLoadGrammar(t *testing.T) {
	language := tree_sitter.NewLanguage(tree_sitter_tree_sitter.Language())
	if language == nil {
		t.Errorf("Error loading TreeSitter grammar")
	}
}
