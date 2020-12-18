include("../DiffusionModel.jl")
include("../Blocking.jl")
using LightGraphs;
using Test;
using GLPK;

@testset "All Tests" begin

@testset "Propogation Test: Low Threshold" begin    
    my_graph_1 = path_graph(5)
    model = DiffusionModel.MultiDiffusionModelConstructor(my_graph_1, 2, [UInt32(1), UInt32(1)])
    model.nodeStates[1][1] = 1
    model.nodeStates[1][2] = 2

    @testset "Basic Diffusion" begin

        DiffusionModel.iterate!(model)
        summary = DiffusionModel.getStateSummary(model)

        @test summary[2] == 2
        @test summary[3] == 3
        
        DiffusionModel.full_run(model)
        summary = DiffusionModel.getStateSummary(model)
    
        @test summary[2] == 5
        @test summary[3] == 5
    
        for node in vertices(model.network)
            for i=1:2
                @test  model.nodeStates[node][i] == 1
            end
        end

    end 

    @testset "Diffusion with Blocking" begin
        
        seeds = Vector([Set([1, 2]), Set([1])])
        DiffusionModel.set_initial_conditions!(model, seeds)

        blockers = Vector([Set([3]), Set()])
        DiffusionModel.set_blocking!(model, blockers)

        DiffusionModel.full_run(model)
        summary = DiffusionModel.getStateSummary(model)

        @test summary[2] == 2
        @test summary[3] == 1

    end

end


@testset "Propogation Test: Higher Threshold" begin
    my_graph_1 = path_graph(5)
    model = DiffusionModel.MultiDiffusionModelConstructor(my_graph_1, 2, [UInt32(2), UInt32(1)])
    model.nodeStates[1][1] = 1
    full_run_1 = DiffusionModel.full_run(model)
    summary = DiffusionModel.getStateSummary(model)
    @test summary[2] == 1
    for node in vertices(model.network)
        for i=1:2
            if node != 1
                @test  model.nodeStates[node][i] == 0
            end
        end
    end
end







@testset "DiffusionModel Test: Binary Tree Graph" begin
    my_graph_1 = binary_tree(4)
    model = DiffusionModel.MultiDiffusionModelConstructor(my_graph_1, 2, [UInt32(1), UInt32(1)])
    model.nodeStates[1][1] = 1
    model.nodeStates[1][2] = 1

    DiffusionModel.full_run(model)

    @summary[2] == nv(my_graph_1)
    @summary[3] == nv(my_graph_1)
    

end

@testset "Diffusion Test: Star Graph" begin
    graph_4 = star_graph(5)
    model = DiffusionModel.MultiDiffusionModelConstructor(graph_4, 2, [UInt32(1), UInt32(1)])
    model.nodeStates[1][1] = 1
    model.nodeStates[1][2] = 1

    DiffusionModel.full_run(model)

    @summary[2] == nv(graph_4)
    @summary[3] == nv(graph_4)

    

end


end