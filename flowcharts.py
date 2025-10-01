import schemdraw
import schemdraw.flow as flow
from schemdraw.elements import DotDotDot

with schemdraw.Drawing(font='Times New Roman', fontsize=14, aspect=1.2, unit=2.0, linewidth=0.8) as d:
    # --- Start and Initialization ---
    start = flow.Start().label("Start")
    d += start

    d += flow.Arrow().down(0.7)
    init = flow.Box(w=6, h=0.9).label("Initialize population $P_0$")
    d += init

    d += flow.Arrow().down(0.7)
    evaluate = flow.Box(w=6, h=0.9).label("Evaluate fitness of $P_t$")
    d += evaluate

    d += flow.Arrow().down(0.7)
    check = flow.Decision(w=6, h=2.5, E="No", S="Yes").label("Max generations reached?")
    d += check

    # --- YES → End ---
    d += flow.Arrow().down(0.7).at(check.S)
    end = flow.Terminal().label("End:\nReturn non-dominated solutions")
    d += end

    # --- NO → Continue Evolution ---
    d += flow.Line().right(6.0).at(check.E)
    d += flow.Arrow().down(0.7)

    offspring = flow.Box(w=6, h=1).label("Generate offspring $Q_t$:\nSelection, Crossover, Mutation")
    d += offspring

    d += flow.Arrow().down(0.7)
    evaluate_offspring = flow.Box(w=6, h=0.9).label("Evaluate fitness of $Q_t$")
    d += evaluate_offspring

    d += flow.Arrow().down(0.7)
    merge = flow.Box(w=6, h=0.9).label("Combine: $R_t = P_t \\cup Q_t$")
    d += merge

    d += flow.Arrow().down(0.7)
    sort = flow.Box(w=6, h=0.9).label("Non-dominated sorting")
    d += sort

    d += flow.Arrow().down(0.7)
    crowd = flow.Box(w=6, h=0.9).label("Crowding distance (per front)")
    d += crowd

    d += flow.Arrow().down(0.7)
    select = flow.Box(w=6, h=0.9).label("Select next generation $P_{t+1}$")
    d += select

    # --- Loop Back ---
    d += flow.Line().left(10).at(select.W)
    d += flow.Line().up(12.5)
    d += flow.Arrow().right(0.1).to(evaluate.N)

    d.save("nsga2_flowchart_final_clean.pdf")

with schemdraw.Drawing(font='Times New Roman', fontsize=9, aspect=1.2, unit=2.0, linewidth=0.8) as d:  # PPO
    start = flow.Start().label("Start")
    d += start

    d += flow.Arrow().down(0.7)
    init = flow.Box(w=6, h=0.9).label("Initialize: Policy Parameters,\n Value Function Parameters")
    d += init

    d += flow.Arrow().down(0.7)
    rollout = flow.Box(w=6, h=0.9).label("Collect Trajectories\n by running policy in the environment")
    d += rollout

    d += flow.Arrow().down(0.7)
    advantages = flow.Box(w=6, h=0.9).label("Compute Rewards and Advantages")
    d += advantages

    d += flow.Arrow().down(0.7)
    actor_update = flow.Box(w=6, h=1.1).label("Update Policy\n by maximizing the PPO-clip objective")
    d += actor_update

    d += flow.Arrow().down(0.7)
    critic_update = flow.Box(w=6, h=0.9).label("Fit and Update Value Function")
    d += critic_update

    # Decision block
    d += flow.Arrow().down(0.7)
    decision = flow.Decision(w=5, h=1.1, E="No", S="Yes").label("Max iterations\nreached?").fill("white")
    d += decision

    # --- YES path: down from Decision to End ---
    d += flow.Arrow().down(0.7).at(decision.S)
    end = flow.Terminal().label("End")
    d += end

    # --- NO path: right and loop back to top of rollout ---
    d += flow.Line().right(1.5).at(decision.E)
    d += flow.Line().up(6)  # adjust height based on spacing
    d += flow.Arrow().left(4.0)  # adjust width to align with rollout
    # d += flow.Arrow().down(0.7)  # arrow into the top of rollout

    # Save the finished figure
    d.save("ppo_flowchart_final.pdf")

with schemdraw.Drawing(unit=1.0, aspect=1.2, fontsize=10) as d:
    # Layout anchors
    x_center = 0
    spacing_x = 5.5
    spacing_y = 2.4

    x_left = x_center - spacing_x
    x_right = x_center + spacing_x

    y_paper = 0
    y_domains = y_paper - spacing_y
    y_split = y_domains - spacing_y
    y_branch = y_split - spacing_y
    y_detail = y_branch - spacing_y

    # 2. Research Domains
    d += flow.Box(w=4.0, h=1.2).at((x_center-2, y_domains)).label("Input Features").fill("#E3F2FD")
    d += flow.Line().at((x_center, y_domains-0.6)).to((x_center, y_split))

    # 3. Fan-out arrows to branches
    d += flow.Arrow().at((x_center, y_split)).to((x_left, y_branch))
    d += flow.Arrow().at((x_center, y_split)).to((x_center, y_branch))
    d += flow.Arrow().at((x_center, y_split)).to((x_right, y_branch))

    # 4. V2X Branch
    d += flow.Box(w=4.0, h=1.2).at((x_left-2, y_branch)).label("Feature Tokenizer 1").fill("#C8E6C9")
    d += flow.Arrow().at((x_left, y_branch - 1.2)).to((x_left, y_detail))
    d += flow.Box(w=4.0, h=1.2).at((x_left, y_detail)).label("[25-44]").fill("#E8F5E9")

    # 5. MTL Branch
    d += flow.Box(w=4.0, h=1.2).at((x_center, y_branch)).label("Feature Tokenizer 2").fill("#FFF9C4")
    d += flow.Arrow().at((x_center, y_branch-1.2)).to((x_center, y_detail))
    d += flow.Box(w=4.0, h=1.2).at((x_center, y_detail)).label("[16-19] - [45-52]").fill("#FFFDE7")

    # 6. Pruning Branch
    d += flow.Box(w=4.0, h=1.2).at((x_right, y_branch)).label("Feature Tokenizer 3").fill("#FFCDD2")
    d += flow.Arrow().at((x_right, y_branch - 1.2)).to((x_right, y_detail))
    d += flow.Box(w=4.0, h=1.2).at((x_right, y_detail)).label("[53-64]").fill("#FFEBEE")

    # Save the clean, aligned result
    d.save("mt-ft.pdf")

with schemdraw.Drawing(unit=1.0, aspect=1.2, fontsize=11) as d:
    # Layout anchors
    x_center = 0
    spacing_x = 5.5
    spacing_y = 2.4

    x_left = x_center - spacing_x
    x_right = x_center + spacing_x

    y_paper = 0
    y_domains = y_paper - spacing_y
    y_split = y_domains - spacing_y
    y_branch = y_split - spacing_y
    y_detail = y_branch - spacing_y
    y_shared = y_detail - spacing_y  # New: y-coordinate for shared block

    # Input Features
    d += flow.Box(w=4.0, h=1.2).at((x_center - 2, y_domains)).label("Input Features").fill("#E3F2FD")
    d += flow.Line().at((x_center, y_domains - 0.6)).to((x_center, y_split))

    # Fan-out arrows to branches
    d += flow.Arrow().at((x_center, y_split)).to((x_left, y_branch))
    d += flow.Arrow().at((x_center, y_split)).to((x_center, y_branch))
    d += flow.Arrow().at((x_center, y_split)).to((x_right, y_branch))

    # Branch 1 (Left)
    d += flow.Box(w=4.0, h=1.2).at((x_left - 2, y_branch)).label("Feature Tokenizer 1").fill("#C8E6C9")
    d += flow.Arrow().at((x_left, y_branch - 1.2)).to((x_left, y_detail))

    # Branch 2 (Center)
    d += flow.Box(w=4.0, h=1.2).at((x_center, y_branch)).label("Feature Tokenizer 2").fill("#FFF9C4")
    d += flow.Arrow().at((x_center, y_branch - 1.2)).to((x_center, y_detail))

    # Branch 3 (Right)
    d += flow.Box(w=4.0, h=1.2).at((x_right, y_branch)).label("Feature Tokenizer N").fill("#FFCDD2")
    d += flow.Arrow().at((x_right, y_branch - 1.2)).to((x_right, y_detail))

    d += DotDotDot().at((x_center + 1.8, y_branch - 0.6)).theta(0)

    # Final Shared Transformer Encoder box (center bottom)
    shared_x = x_center
    shared_box = d.add(flow.Box(w=15.5, h=1.2).at((shared_x, y_shared+2.4)).label("Shared Transformer Encoder").fill("#D1C4E9"))

    # Define y position for output heads
    y_output = y_shared + 1.2
    y_branch2 = y_output - spacing_y
    y_detail2 = y_branch2 - spacing_y

    d += flow.Arrow().at((x_center, y_output)).to((x_left, y_branch2))
    d += flow.Arrow().at((x_center, y_output)).to((x_center, y_branch2))
    d += flow.Arrow().at((x_center, y_output)).to((x_right, y_branch2))

    d += flow.Box(w=4.0, h=1.2).at((x_left - 2, y_branch2)).label("Output Head 1").fill("#C8E6C9")

    # Branch 2 (Center)
    d += flow.Box(w=4.0, h=1.2).at((x_center - 2, y_branch2)).label("Output Head 2").fill("#FFF9C4")

    # Branch 3 (Right)
    d += flow.Box(w=4.0, h=1.2).at((x_right - 2, y_branch2)).label("Output Head N").fill("#FFCDD2")

    d += DotDotDot().at((x_center + 1.8, y_branch2 - 0.6)).theta(0)

    # Save the figure
    d.save("mt-ft_model.pdf")
