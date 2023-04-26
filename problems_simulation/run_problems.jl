include("../problems_simulation/cryingBabyProblem_solvers.jl")
include("../problems_simulation/miniHallwayProblem_solvers.jl")
include("../problems_simulation/PaintProblem_solvers.jl")
include("../problems_simulation/queryProblem_solvers.jl")
include("../problems_simulation/rockSampleProblem_solvers.jl")
include("../problems_simulation/tigerProblem_solvers.jl")
include("../problems_simulation/tMazeProblem_solvers.jl")





run_baby_solvers(false,1000,10)
run_hallway_solvers(false, 1000,10)
run_paint_solvers(false, 1000,10)
run_query_solvers(false, 1000,10)
run_rock_solvers(false, 1000,10)
run_tiger_solvers(false, 1000,10)
run_tMaze_solvers(false, 1000,10)