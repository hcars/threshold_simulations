include("../DiffusionModel.jl")
include("../Blocking.jl")
using LightGraphs;
using Test;
using GLPK;

@testset "All Tests" begin

@testset "Propogation Test: Low Threshold" begin    
    my_graph_1 = path_graph(5)
    node_states_1 = Dict{Int,UInt}()
    get!(node_states_1, 1, 1)
    blockedDict_1 = Dict{Int,UInt}()
    thresholdStates_1 = Dict{Int,UInt32}()
    model = DiffusionModel.MultiDiffusionModel(my_graph_1, node_states_1, thresholdStates_1, blockedDict_1, [UInt32(1), UInt32(1)], UInt32(0))
    DiffusionModel.iterate!(model)
    summary = DiffusionModel.getStateSummary(model)
    @test summary[2] == 2
    DiffusionModel.full_run(model)
    summary = DiffusionModel.getStateSummary(model)
    @test summary[2] == 5
    for node in vertices(model.network)
        @test get(model.nodeStates, node, 0) == 1
    end
end


@testset "Propogation Test: Higher Threshold" begin
    # Define Graph
    my_graph_1 = path_graph(5)
    # Restate and try with higher threshold 
    node_states_1 = Dict{Int,UInt}()
    get!(node_states_1, 1, 1)
    blockedDict_1 = Dict{Int,UInt}()
    thresholdStates_1 = Dict{Int,UInt32}()
    model = DiffusionModel.MultiDiffusionModel(my_graph_1, node_states_1, thresholdStates_1, blockedDict_1, [UInt32(2), UInt32(1)], UInt32(0))
    full_run_1 = DiffusionModel.full_run(model)
    summary = DiffusionModel.getStateSummary(model)
    @test summary[2] == 1
    for node in vertices(model.network)
        if node != 1
            @test get(model.nodeStates, node, 0) == 0
        end
    end
end

@testset "Propogation Test 3" begin
    my_graph_1 = path_graph(5)
    node_states_1 = Dict{Int,UInt}()
    get!(node_states_1, 1, 1)
    blockedDict_1 = Dict{Int,UInt}()
    thresholdStates_1 = Dict{Int,UInt32}()
    model = DiffusionModel.MultiDiffusionModel(my_graph_1, node_states_1, thresholdStates_1, blockedDict_1, [UInt32(1), UInt32(1)], UInt32(0))
    summary = DiffusionModel.getStateSummary(model)
    full_run_1 = DiffusionModel.full_run(model)
    summary = DiffusionModel.getStateSummary(model)
    @test summary[2] == 5
    for node in vertices(model.network)
        @test get(model.nodeStates, node, 0) == 1
    end
    for i = 1:length(full_run_1)
        updates = full_run_1[i]
        @test length(keys(updates[1])) == 1
    end
end


@testset "Diffusion Test Test: Block on Path Graph" begin
    my_graph_1 = path_graph(5)
    add_vertex!(my_graph_1)
    add_edge!(my_graph_1, 6, 3)
    node_states_1 = Dict{Int,UInt}()
    get!(node_states_1, 1, 1)
    get!(node_states_1, 6, 1)
    blockedDict_1 = Dict{Int,UInt}()
    thresholdStates_1 = Dict{Int,UInt32}()
    model = DiffusionModel.MultiDiffusionModel(my_graph_1, node_states_1, thresholdStates_1, blockedDict_1, [UInt32(1), UInt32(1)], UInt32(0))
    full_run_1 = DiffusionModel.full_run(model)
    
    
end


@testset "Diffusion Test: A simple graph that I created" begin
    # Build graph
    my_graph_2 = SimpleGraph()
    add_vertices!(my_graph_2, 8)
    add_edge!(my_graph_2, 1, 2)
    add_edge!(my_graph_2, 1, 4)
    add_edge!(my_graph_2, 2, 3)
    add_edge!(my_graph_2, 4, 3)
    add_edge!(my_graph_2, 3, 6)
    add_edge!(my_graph_2, 3, 7)
    add_edge!(my_graph_2, 7, 8)
    add_edge!(my_graph_2, 7, 6)
    add_edge!(my_graph_2, 6, 5)
    add_edge!(my_graph_2, 5, 8)
    node_states_2 = Dict{Int,UInt}()
    get!(node_states_2, 2, 1)
    get!(node_states_2, 4, 1)
    get!(node_states_2, 5, 2)
    get!(node_states_2, 7, 2)
    blockedDict_2 = Dict{Int,UInt}()
    thresholdStates_2 = Dict{Int,UInt32}()
    model_other = DiffusionModel.MultiDiffusionModel(my_graph_2, node_states_2, thresholdStates_2, blockedDict_2, [UInt32(2), UInt32(2)], UInt32(0))
   
end


@testset "DiffusionModel Test: Binary Tree Graph" begin
    graph_3 = binary_tree(4)
    node_states_3 = Dict(1 => 3)
    blockedDict_3 = Dict{Int,UInt}()
    thresholdStates_3 = Dict{Int,UInt32}()

    model_3 = DiffusionModel.MultiDiffusionModel(graph_3, node_states_3, thresholdStates_3, blockedDict_3, [UInt32(1), UInt32(1)], UInt32(0))
    full_run_3 = DiffusionModel.full_run(model_3)

    

end

@testset "Diffusion Test: Star Graph" begin
    graph_4 = star_graph(5)
    node_states_4 = Dict(2 => 3)
    blockedDict_4 = Dict{Int,UInt}()
    thresholdStates_4 = Dict{Int,UInt32}()

    model_4 = DiffusionModel.MultiDiffusionModel(graph_4, node_states_4, thresholdStates_4, blockedDict_4, [UInt32(1), UInt32(1)], UInt32(0))
    full_run_4 = DiffusionModel.full_run(model_4)

    

end


end