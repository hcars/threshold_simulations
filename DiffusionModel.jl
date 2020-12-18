module DiffusionModel


	using LightGraphs;





	mutable struct MultiDiffusionModel 
		network::SimpleGraph
		nodeStates::Matrix{UInt1}
		thresholds::Matrix{UInt}
		blockedDict::Dict{Int, Vector}
		θ_i :: Vector{UInt32}
		t::UInt32
	end

	function MultiDiffusionModelConstructor(graph, num_infections::Int)
		nodeStates = zeros(Int, num_infections, nv(graph))
		thresholds = ones(Int, num_infections, nv(graph))
		blockedDict = Dict{Int, Vector}()
		return MultiDiffusionModel(graph, nodeStates, thresholds, blockedDict, [UInt32(2), UInt32(2)], UInt(0))
	end

	function MultiDiffusionModelConstructor(graph, num_infections::Int, θ_i::Vector{UInt32})
		nodeStates = zeros(Int, num_infections, nv(graph))
		thresholds = ones(Int, num_infections, nv(graph))
		blockedDict = Dict{Int, Vector}()
		return MultiDiffusionModel(graph, nodeStates, thresholds, blockedDict, θ_i, UInt(0))
	end

	function set_initial_conditions!(model::MultiDiffusionModel, num_infections::Int, seeds::Set{Int})
		nodeStates = zeros(Int, num_infections, nv(graph))
		model.blockedDict =  Dict{Int, Vector}()
		for seed in seeds
			infection = rand(1:num_infections)
			nodeStates[seed][infection] = 1
		end
		model.t = UInt32(0)
		model.nodeStates = nodeStates
	end

	function set_initial_conditions!(model::MultiDiffusionModel, seeds::Tuple{Set{Int}, Set{Int}})
		nodeStates = Dict{Int, UInt}()
		model.blockedDict =  Dict{Int, Vector}()
		for i=1:length(seeds)
			for seed in seeds[i]
				nodeStates[seed][i] = 1
				
			end
		end
		model.t = UInt32(0)
		model.nodeStates = nodeStates
	end

	function set_blocking!(model::MultiDiffusionModel, blockers::Vector)
		blockingDict = Dict{Int, Vector}()
		for i=1:length(blockers)
			curr_set = blockers[i]
			for node in curr_set
				state = get(blockingDict, node, zeros(UInt1, size(model.nodeStates, 1))
				state[i] = 1
			end
		end
		model.blockedDict = blockingDict
	end

	function iterate!(model::MultiDiffusionModel)::Vector
		"""
		This completes a one time step update.
		"""
		num_infections =size(model.nodeStates, 1)
		updated = [Dict{Int, UInt32}() for i=1:num_infections]
		for u in vertices(model.network)
			u_state = model.nodeStates[u]
			cnt_infected = zeros(UInt, num_infections)
					for v in all_neighbors(model.network, u)
						for infection=1:num_infections 	
							cnt_infected[infection] += model.nodeStates[v][infection]
						end
					end
					for infection=1:num_infections
						old_state = u_state[infection]
						if (cnt_infected[infection] >= model.thresholds[u][infection]) && (get(model.blockedDict, u, 0)[infection] != 1)
							u_state[infection] = 1
							if old_state != 1
								get!(updated[infection], u, cnt_infected[infection])
							end
						end
					end
		end
		for i=1:num_infections
			for u in keys(updated[infection])
				model.nodeStates[u][i] = 1
			end
		end

		model.t += 1
		return updated
	end
							
							
	function getStateSummary(model::MultiDiffusionModel)
		num_infections = size(model.nodeStates, 1)
		state_summary = zeros(Int, num_infections + 1)
		for u in vertices(model.network)
			for i=1:num_infections
				state_summary[num_infections] += model.nodeStates[u][num_infections]
			end
		end
		state_summary[1] = nv(model.network) - sum(state_summary[2:length(state_summary)])
		return state_summary
	end



	function full_run(model::MultiDiffusionModel)::Vector
		updates = Vector()
		updated = iterate!(model)
		max_infections = nv(model.network)
		append!(updates, [updated])
		iter_count = 0
		all_empty = false
		while !(all_empty) && (iter_count < max_infections)
			updated = iterate!(model)
			all_empty = true
			for update in updated
				if !(isempty(update))
					all_empty = false
				end
			end
			if !all_empty
				append!(updates, [updated])
			end
			iter_count += 1
		end
		return updates
	end


end
