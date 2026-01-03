#include <gtest/gtest.h>
#include "task_graph.h"
#include <random>
#include <set>
#include <algorithm>

using namespace mini_image_pipe;

// Feature: mini-image-pipe, Property 11: DAG Cycle Detection
// Validates: Requirements 5.1
TEST(TaskGraphPropertyTest, CycleDetection) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> nodeDist(3, 10);
    
    for (int iter = 0; iter < 100; iter++) {
        TaskGraph graph;
        int numNodes = nodeDist(gen);
        
        // Add nodes
        for (int i = 0; i < numNodes; i++) {
            graph.addTask("Task" + std::to_string(i), nullptr);
        }
        
        // Add random valid edges (forward only to ensure DAG)
        std::uniform_int_distribution<int> edgeDist(0, numNodes - 1);
        for (int i = 0; i < numNodes * 2; i++) {
            int from = edgeDist(gen);
            int to = edgeDist(gen);
            
            // Only add forward edges
            if (from < to) {
                graph.addDependency(from, to);
            }
        }
        
        // Graph should be valid (no cycles)
        EXPECT_TRUE(graph.validate());
        
        // Now try to add a back edge that would create a cycle
        // Pick two nodes where there's a path from A to B, then try to add B->A
        std::vector<int> order = graph.getTopologicalOrder();
        if (order.size() >= 2) {
            int first = order[0];
            int last = order[order.size() - 1];
            
            // Try to add edge from last to first (would create cycle if path exists)
            bool added = graph.addDependency(last, first);
            
            // If there was a path from first to last, this should fail
            // The graph should still be valid after rejection
            EXPECT_TRUE(graph.validate());
        }
    }
}

// Feature: mini-image-pipe, Property 12: Dependency Ordering
// Validates: Requirements 5.2, 5.4, 5.6
TEST(TaskGraphPropertyTest, DependencyOrdering) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> nodeDist(3, 15);
    
    for (int iter = 0; iter < 100; iter++) {
        TaskGraph graph;
        int numNodes = nodeDist(gen);
        
        // Add nodes
        for (int i = 0; i < numNodes; i++) {
            graph.addTask("Task" + std::to_string(i), nullptr);
        }
        
        // Add random forward edges
        std::uniform_int_distribution<int> edgeDist(0, numNodes - 1);
        std::set<std::pair<int, int>> edges;
        
        for (int i = 0; i < numNodes * 2; i++) {
            int from = edgeDist(gen);
            int to = edgeDist(gen);
            
            if (from < to) {
                if (graph.addDependency(from, to)) {
                    edges.insert({from, to});
                }
            }
        }
        
        // Get topological order
        std::vector<int> order = graph.getTopologicalOrder();
        EXPECT_EQ(order.size(), numNodes);
        
        // Verify: for every edge (from, to), 'from' appears before 'to' in order
        std::vector<int> position(numNodes);
        for (int i = 0; i < numNodes; i++) {
            position[order[i]] = i;
        }
        
        for (const auto& edge : edges) {
            EXPECT_LT(position[edge.first], position[edge.second])
                << "Dependency violation: " << edge.first 
                << " should come before " << edge.second;
        }
    }
}

// Test for independent task detection
TEST(TaskGraphPropertyTest, IndependentTaskDetection) {
    std::random_device rd;
    std::mt19937 gen(rd());
    
    for (int iter = 0; iter < 100; iter++) {
        TaskGraph graph;
        
        // Create a simple graph with known independent tasks
        // A -> B -> C
        // D -> E
        // A and D are independent, B and D are independent, etc.
        
        int a = graph.addTask("A", nullptr);
        int b = graph.addTask("B", nullptr);
        int c = graph.addTask("C", nullptr);
        int d = graph.addTask("D", nullptr);
        int e = graph.addTask("E", nullptr);
        
        graph.addDependency(a, b);
        graph.addDependency(b, c);
        graph.addDependency(d, e);
        
        // A and D should be independent
        EXPECT_TRUE(graph.areIndependent(a, d));
        EXPECT_TRUE(graph.areIndependent(d, a));
        
        // B and D should be independent
        EXPECT_TRUE(graph.areIndependent(b, d));
        EXPECT_TRUE(graph.areIndependent(d, b));
        
        // A and B should NOT be independent (A -> B)
        EXPECT_FALSE(graph.areIndependent(a, b));
        
        // A and C should NOT be independent (A -> B -> C)
        EXPECT_FALSE(graph.areIndependent(a, c));
    }
}
