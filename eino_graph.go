package fileprocessor

// This file demonstrates how to wire a [Retriever] into an eino Graph
// pipeline. It is not imported by default — callers who want Graph
// composability import "github.com/cloudwego/eino/compose" and build
// the graph themselves.
//
// Example: RAG pipeline (retrieve → rerank → format)
//
//	retriever, _ := fileprocessor.NewRetriever(ctx, &fileprocessor.RetrieverConfig{
//	    Store:    vecStore,
//	    Chunks:   chunkStore,
//	    Embedder: embedder,
//	    Keyword:  keywordIndex,
//	    ExpandParent: true,
//	})
//
//	graph := compose.NewGraph[string, string]()
//
//	// AddRetrieverNode wraps the Retriever as a graph node.
//	// The node accepts a query string and returns []*schema.Document.
//	graph.AddRetrieverNode("retrieve", retriever,
//	    compose.WithOutputKey("docs"),
//	)
//
//	// Add a Lambda node for reranking or formatting.
//	graph.AddLambdaNode("format", compose.InvokableLambda(
//	    func(ctx context.Context, docs []*schema.Document) (string, error) {
//	        // Format results for LLM context.
//	        return FormatResults(docs, ""), nil
//	    },
//	), compose.WithInputKey("docs"))
//
//	// Wire the edges: retrieve → format
//	graph.AddEdge("retrieve", "format")
//
//	// Compile and invoke.
//	chain, _ := graph.Compile(ctx)
//	result, _ := chain.Invoke(ctx, "what is the timeline?")
//
// Graph benefits:
//   - Callbacks fire on each node (langfuse, langsmith, OpenTelemetry)
//   - Nodes run in parallel when connected via AddBranch
//   - State can be shared via WithGenLocalState
//   - Streaming is supported via StreamInvoke
//
// For more complex pipelines (e.g., parallel retrieval from multiple
// knowledge bases), use compose.NewWorkflow or compose.NewGraph with
// branches:
//
//	graph.AddRetrieverNode("kb1", retrieverKB1)
//	graph.AddRetrieverNode("kb2", retrieverKB2)
//	graph.AddBranch("merge", compose.NewConditionalBranch(
//	    func(ctx context.Context, input string) ([]string, error) {
//	        return []string{"kb1", "kb2"}, nil  // run both
//	    },
//	    map[string]compose.GraphAddNodeOpt{
//	        "kb1": {},
//	        "kb2": {},
//	    },
// ))
