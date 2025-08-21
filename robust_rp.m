%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Robust Utility Reconstruction Simulation
% Author: Luke Snow, Aug 21 2025
%
% This script implements a Monte Carlo simulation of utility reconstruction
% under noisy multi-agent decision data, comparing the **classical Afriat
% revealed preference (RP) test** with a **distributionally robust
% extension** (finite-reduction algorithm).
%
% -------------------------------------------------------------------------
% PURPOSE:
%   - Generate synthetic decision data from agents solving multi-objective
%     optimization problems subject to random probe constraints.
%   - Corrupt these observed responses with controlled Gaussian noise.
%   - Apply two reconstruction methods:
%       (1) Classical Afriat inequalities (linear program).
%       (2) Robust Afriat formulation (finite-reduction iterative program).
%   - Evaluate how well reconstructed utilities explain out-of-sample test
%     probes, using Hausdorff-like set distances between predicted and true
%     response surfaces.
%   - Quantify and compare the **worst-case prediction error** and the
%     **average error** of each method as a function of noise variance.
%
% -------------------------------------------------------------------------
% STRUCTURE:
%   1. Parameters and pre-allocation
%      - Monte Carlo repetitions (MC_out, MC_in).
%      - Noise variance sweep (rg_space).
%      - Cases: no noise, noisy classical RP, noisy robust RP.
%
%   2. Data Generation
%      - For each probe, solve a ground-truth optimization problem with
%        concave utilities to obtain agent responses B(:,:,k).
%
%   3. Scaling
%      - Normalize problem data for conditioning.
%
%   4. Reconstruction
%      - Case n=1 (no noise): trivial check.
%      - Case n=2 (classical Afriat): solve LP with Afriat inequalities.
%      - Case n=3 (robust Afriat): iterative finite-reduction method.
%
%   5. Utility Surface Construction
%      - Reconstruct utilities on a grid via Afriat lower envelope.
%
%   6. Prediction / Testing
%      - Solve test optimizations with reconstructed utilities.
%      - Compare to true optimizations using Hausdorff-like distances.
%
%   7. Post-Processing
%      - Compute statistics (max error, average error) across MC batches.
%      - Plot error vs. noise level, comparing classical vs robust methods.
%      - Track robust method’s constraint violation across iterations.
%
% -------------------------------------------------------------------------
% OUTPUTS:
%   - Figures:
%       (a) Average worst-case prediction error vs. noise variance.
%       (b) Average prediction error vs. noise variance.
%   - Printed statistics (mean and std of errors across runs).
%   - CV_st: trajectory of constraint violation in the robust method.
%
% -------------------------------------------------------------------------
% NOTES:
%   - The ground-truth generating utilities are fixed nonlinear forms
%     (quadratic, square-root, etc.).
%   - Reconstruction solves optimization problems repeatedly, so runtime
%     may be significant for large MC_out/MC_in.
%   - All helper functions are defined at the bottom of this file.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%clear; clc;
close all;
clear std;

tic;

% Outer and inner Monte-Carlo resolutions. Increase for higher accuracy:
MC_out = 5; % MC over maxes  (outer Monte Carlo repetitions across random seeds)
MC_in  = 5; % max over difts  (inner repetitions used for max/avg statistics)

rg_space = [0.1,0.5,1,2,3];   % noise scale settings to sweep (sqrt used later)
%rg_space = [2];

% nspace encodes three cases:
%   1) 0 -> no noise
%   2) 1 -> noise with classical (Afriat LP) reconstruction
%   3) 1 -> noise with robust (finite-reduction) reconstruction
% The loop index n determines which branch (classical vs robust) is used.
nspace = [0,1,1]; % no-noise, noise w/ classic utility estimate, noise w/ robust utility estimate

% Storage for evaluation metrics:
%   dift:    per (outer, inner, n-case, noise-level) average distance measure (Hausdorff-like)
%   met:     per (outer, inner, n-case, noise-level) optimization metric (e.g., LP objective)
%   CV_st:   constraint-violation trajectory for robust procedure across iterations
dift   = zeros(MC_out,MC_in,length(nspace),length(rg_space));
met    = dift;
CV_st  = zeros(MC_out,MC_in,100,length(rg_space));

L1dist_ = zeros(MC_out,MC_in,length(nspace),length(rg_space)); % (not used downstream but reserved)

% v_sol: stores the robust method’s vector v = [v1,v2] per run (for inspection)
v_sol   = zeros(MC_out,MC_in,length(nspace),length(rg_space),2);

% =========================
% Robust numerical options
% =========================
% Common fmincon options:
%  - Use SQP to avoid ill-conditioned KKT solves in the interior-point method
%  - Scale objectives/constraints automatically for better conditioning
%  - Tight tolerances to stabilize the finite-reduction loop solves
options = optimoptions('fmincon',...
    'Algorithm','sqp', ...
    'ScaleProblem','obj-and-constr', ...
    'Display','off', ...
    'ConstraintTolerance',1e-8, ...
    'OptimalityTolerance',1e-8, ...
    'StepTolerance',1e-10, ...
    'MaxFunctionEvaluations',5e4);

% Use linprog for the classical Afriat inequalities:
%  - dual-simplex is robust for dense, moderately-sized LPs
%  - suppress console output
lpoptions = optimoptions('linprog','Display','none','Algorithm','dual-simplex','ConstraintTolerance',1e-10);

% =========================
% Monte Carlo loops
% =========================
for mc_out = 1:MC_out
    for mc_in = 1:MC_in
        for n=1:length(nspace)
            for rg = 1:length(rg_space)

                contour_res = 40; % (unused here, kept for compatibility with prior code)
                N = 5;           % number of probe–response observations in training
                M = 3;           % number of agents

                % Tensor B holds agent decisions for each probe: size 2 x M x N
                % (two-dimensional action per agent, e.g., beta components)
                B  = zeros(2,M,N);
                A  = zeros(2,N);            % probes (constraints) per time k
                y  = zeros(1,N);            % budgets (here fixed to 1)

                % observation noise applied to B:
                % note the scale is sqrt(rg_space(rg)), so variance ~ rg_space(rg)
                noise = sqrt(rg_space(rg))*nspace(n)*randn(size(B));

                % =========================
                % Generate training samples
                % =========================
                % For each probe k, we generate a random constraint alpha,
                % then solve a “true” multi-objective program (weighted sum)
                % to produce a ground-truth coordinated response B(:,:,k).
                for k=1:N
                    mu  = [1/3,1/3,1/3];  % equal weights for 3 agents in the ground-truth generator
                    y_  = 1;              % total resource budget
                    alph = [rand+0.1, rand+0.1]; % random positive 2D constraint vector

                    % ground-truth objective (concave utilities; we minimize negative)
                    fcn = @(x) -mu(1)*(x(1,1)^2 * x(2,1)^2) ...
                               -mu(2)*(x(1,2) * x(2,2)^(1/2)) ...
                               -mu(3)*((x(1,3)^(1/2)) * x(2,3));

                    % fmincon plumbing
                    A_   = [];
                    b_   = [];
                    x0   = zeros(2,M); % start at zero
                    Aeq  = [];
                    beq  = [];
                    lb   = zeros(2,M); % nonnegativity on actions
                    ub   = [];

                    % Solve the “true” coordinated response under constraint RPcon2(x, alph, y_, M)
                    [x_, ~] = fmincon(fcn, x0, A_, b_, Aeq, beq, lb, ub, @(x) RPcon2(x, alph, y_, M), options);

                    % Store probe and response
                    B(:,:,k) = x_;
                    A(:,k)   = alph';
                    y(k)     = 1;
                end

                % =========================
                % Scaling for conditioning
                % =========================
                % To improve numerical stability in reconstruction, we scale
                % constraints and observations to unit-ish magnitudes for
                % the optimization problems below (only internal; metrics can use originals).
                A_scale   = max(1, max(abs(A(:))));
                B_scale   = max(1, max(abs(B(:))));
                A_n       = A / A_scale;                     % scaled probe matrix
                B_n       = B / B_scale;                     % (reserved, not used directly below)
                b_sol_n   = max(B + noise, 0.01) / B_scale;  % scaled noisy observations (lower bounded)

                % =========================
                % Solve Afriat / Robust RP
                % =========================
                % We reconstruct utilities via (1) classical Afriat LP or
                % (2) robust finite-reduction (depending on n).
                lambda_sol = zeros(N,M); % dual variables per (t, i)
                u_sol      = zeros(N,M); % utility levels per (t, i)

                if (n <= 2)
                    % -------------------------
                    % Classical Afriat via LP
                    % -------------------------
                    % For each agent independently, we solve an LP that searches
                    % for (u, lambda, phi) satisfying Afriat inequalities:
                    %   u_i - u_j - lambda_j * p_j' (x_i - x_j) + phi <= 0
                    % Here p_j is the probe (scaled) at time j, x_i the observed
                    % (scaled) decision at time i, and phi is maximized (slack).
                    lam_min = 1e-6; % strictly positive lower bound to prevent degeneracy
                    lam_max = 1e6;  % generous upper bound on lambda
                    % u is free (LP bounds keep it finite)

                    for agent = 1:M
                        % Decision vector z = [u(1:N), lambda(1:N), phi]
                        nvar = 2*N + 1;

                        % Maximize phi  <=> minimize -phi
                        f = zeros(nvar,1); f(end) = -1;

                        % Build Aineq z <= bineq for all (i,j) pairs
                        Aineq = zeros(N*N, nvar);
                        bineq = zeros(N*N, 1);
                        row = 0;
                        for i = 1:N
                            for j = 1:N
                                row = row + 1;
                                % u_i - u_j - lambda_j * p_ij + phi <= 0
                                Aineq(row, i)       =  1;   % +u_i
                                Aineq(row, j)       = -1;   % -u_j
                                pij = A_n(:,j)' * (b_sol_n(:,agent,i) - b_sol_n(:,agent,j)); % scaled inner-product
                                Aineq(row, N + j)   = -pij; % - lambda_j * pij
                                Aineq(row, 2*N + 1) =  1;   % +phi
                                bineq(row)          =  0;
                            end
                        end

                        % Variable bounds: u free, lambda in [lam_min, lam_max], phi in [-10, 0]
                        lb = [-inf(N,1); lam_min*ones(N,1); -10];
                        ub = [ inf(N,1); lam_max*ones(N,1);  0];

                        Aeq = []; beq = [];

                        % Solve LP (dual-simplex)
                        [z, fval_, exitflag_, ~] = linprog(f, Aineq, bineq, Aeq, beq, lb, ub, lpoptions);
                        if exitflag_ <= 0
                            warning('Afriat LP did not converge: agent %d', agent);
                        end

                        % Parse solution into u and lambda (phi not stored)
                        u_sol(:,agent)      = z(1:N);
                        lambda_sol(:,agent) = z(N+1:2*N);
                        % phi_sol            = z(end);
                    end

                    % Record LP objective for this configuration (last agent’s value)
                    met(mc_out,mc_in,n,rg) = fval_;

                else
                    % -----------------------------------
                    % Robust Afriat (finite reduction)
                    % -----------------------------------
                    % Iterative procedure:
                    %   (1) maximize worst constraint violation over synthetic B
                    %   (2) finite-reduction step to update (u, lambda, v)
                    % until CV (max violation) <= delta.
                    wr    = 0.5;  % weight for v(2) in robust objective
                    delta = 0.5;  % stopping tolerance on constraint violation
                    v     = [1,1];% robust parameters (v1,v2)
                    psi   = ones(2,M,N).*rand; %#ok<NASGU> % (unused placeholder)
                    B_set = zeros(2,M,N,100); % stores adversarial samples over iterations
                    itr   = 1;                % iteration counter for B_set
                    CV    = 10;               % initialize violation high

                    % initialize primal variables for finite-reduction stage
                    u      = rand(N,M);
                    lambda = 0.5*ones(N,M);
                    v      = rand(1,2);

                    cv_st_it = 1; % index into CV_st storage

                    % Box bounds improving conditioning in the robust step
                    lam_lb = 1e-6*ones(N,M); lam_ub = 1e6*ones(N,M);
                    u_lb   = -1e6*ones(N,M); u_ub   =  1e6*ones(N,M);
                    v_lb   = zeros(1,2);     v_ub   =  1e3*ones(1,2);

                    while (CV > delta)

                        % ---------- Maximum constraint violation subproblem ----------
                        % Maximize:
                        %   mcval(u,lambda,v, A_n, b_sol_n, B_var)
                        % over B_var subject to A'B <= 1 (scaled).
                        B_var = optimvar('B_var',2,M,N,'LowerBound',zeros(2,M,N));
                        obj   = fcn2optimexpr(@mcval, u, lambda, v, A_n, b_sol_n, B_var, N, M);
                        prob  = optimproblem('Objective', obj, 'ObjectiveSense','maximize');

                        % Constraint A'B <= 1 for each sample k (here RHS set to 20 for slack)
                        constraint = fcn2optimexpr(@RPcon3, B_var, A_n, 20*ones(1,N), M, N);
                        prob.Constraints.c1 = constraint <= zeros(N,1);

                        % Solve the adversarial B_var problem
                        p0.B_var = ones(size(B_var));
                        [sol, CV] = solve(prob, p0, 'Options', options);

                        % Record CV trajectory and store the adversarial sample
                        CV_st(mc_out,mc_in,cv_st_it,rg) = CV;
                        cv_st_it = cv_st_it + 1;
                        B_set(:,:,:,itr) = sol.B_var;
                        itr = itr + 1;

                        % If violation remains positive, update (u,lambda,v) by finite reduction
                        if (CV > 0)
                            % ---------- Finite reduction program ----------
                            lambda = optimvar('lambda',N,M,'LowerBound',lam_lb,'UpperBound',lam_ub);
                            u      = optimvar('u',N,M,'LowerBound',u_lb,'UpperBound',u_ub);
                            v      = optimvar('v',1,2,'LowerBound',v_lb,'UpperBound',v_ub);

                            prob = optimproblem('ObjectiveSense','minimize');

                            % Objective: wr*v2 + v1  + small ridge to regularize parameters
                            ridge = 1e-8;
                            prob.Objective = wr*v(2) + v(1) + ridge*(sum(lambda(:).^2) + sum(u(:).^2) + sum(v(:).^2));

                            % Constraint: mcval2(u,lambda,v, A_n, b_sol_n, B_set, ..., itr) <= v1
                            expr1 = fcn2optimexpr(@mcval2, u, lambda, v, A_n, b_sol_n, B_set, N, M, itr);
                            prob.Constraints.c1 = expr1 <= v(1);

                            % Warm starts
                            p0.lambda = ones(size(lambda));
                            p0.u      = ones(size(u));
                            p0.v      = ones(1,2);

                            % Solve finite reduction update
                            [sol_, fval_] = solve(prob, p0, 'Options', options); %#ok<NASGU>

                            % Extract variables for next iteration
                            u      = sol_.u;
                            lambda = sol_.lambda;
                            v_     = sol_.v;
                            v      = sol_.v;
                        end
                    end

                    % Normalization post-processing of (u,lambda)
                    scale_fac  = max(max(max(u,lambda)));
                    u_sol      = u./scale_fac;
                    lambda_sol = lambda./scale_fac;

                    % Store robust v and record feasibility metric
                    v_sol(mc_out,mc_in,n,rg,:) = v_;
                    met(mc_out,mc_in,n,rg)     = af_eval(u_sol,lambda_sol,A_n,B_set,N,M);
                end

                % =========================
                % Reconstruct utilities U
                % =========================
                % Given (u_sol, lambda_sol), build the reconstructed utility
                % over a grid (betaspace) via the Afriat lower envelope:
                %   U_hat(beta) = min_t [ u_t + lambda_t * a_t' * (beta - b_t) ]
                % where a_t and b_t are scaled probe and response.
                for agent=1:M
                    betaspace = linspace(0,2,20); % grid for each component
                    U(:,:,agent)      = zeros(length(betaspace)); % reconstructed
                    U_act             = U(:,:,agent); %#ok<NASGU> % (templates for “true” forms)
                    U_act_tr          = U(:,:,agent); %#ok<NASGU>
                    U_act_cd          = U(:,:,agent); %#ok<NASGU>
                    U_act_cdd         = U(:,:,agent); %#ok<NASGU>

                    for b1 = 1:length(betaspace)
                        for b2 = 1:length(betaspace)
                            % initialize with t=1
                            min_ = u_sol(1,agent) + lambda_sol(1,agent)*A_n(:,1)'*([betaspace(b1);betaspace(b2)] - b_sol_n(:,agent,1));
                            U(b1,b2,agent) = min_;
                            % take min over all t
                            for t=2:length(u_sol(:,1))
                                eval = u_sol(t,agent) + lambda_sol(t,agent)*A_n(:,t)'*([betaspace(b1);betaspace(b2)] - b_sol_n(:,agent,t));
                                if (eval < min_)
                                    min_ = eval;
                                    U(b1,b2,agent) = eval;
                                end
                            end
                            % (Optional) true benchmark utilities, unused in metrics
                            U_act(b1,b2)     = (betaspace(b1)*betaspace(b2))^2;
                            U_act_tr(b1,b2)  = betaspace(b1) + betaspace(b2);
                            U_act_cd(b1,b2)  = (betaspace(b1))^(1/4) + betaspace(b2);
                            U_act_cdd(b1,b2) = betaspace(b1) + (betaspace(b2))^(1/4);
                        end
                    end
                end

                % =========================
                % Prediction / Testing
                % =========================
                % Evaluate predictive performance on T random test probes by:
                %  - solving for predicted coordinated response under reconstructed U
                %  - solving for true response under ground-truth objective
                %  - computing the Hausdorff-distance (https://en.wikipedia.org/wiki/Hausdorff_distance) between surfaces
                T   = 10;
                B_  = zeros(2,M,T); % (unused)
                Bu_ = B_;
                A_  = zeros(2,T);   % (unused)

                HD = zeros(1,T);    % per test probe distance measure
                for k=1:T
                    mu   = [1/3,1/3,1/3];
                    y_   = 1;
                    alph = [rand+0.1, rand+0.1]; % random test probe

                    % True objective again (for comparison surface)
                    fcn  = @(x) -mu(1)*(x(1,1)^2 * x(2,1)^2) ...
                                -mu(2)*(x(1,2) * x(2,2)^(1/2)) ...
                                -mu(3)*((x(1,3)^(1/2)) * x(2,3));

                    % fmincon plumbing for test solve
                    A__  = [];
                    b__  = [];
                    x0   = 0.1 + rand(2,M)*0.4;  % safer init (avoid exact zero)
                    Aeq  = [];
                    beq  = [];
                    lb   = 1e-3*ones(2,M);
                    ub   = [];

                    % Reconstructed objective proxy via grid lookup of U
                    fcn2 = @(x) - mu(1)*U(min(ceil(x(1,1)*10),20),min(ceil(x(2,1)*10),20),1) ...
                                - mu(2)*U(min(ceil(x(1,2)*10),20),min(ceil(x(2,2)*10),20),2) ...
                                - mu(3)*U(min(ceil(x(1,3)*10),20),min(ceil(x(2,3)*10),20),3);

                    % Predicted “reconstructed-utility” optimal action
                    [x__, ~] = fmincon(fcn2, x0, A__, b__, Aeq, beq, lb, ub, @(x) RPcon2(x,alph,y_,M), options);
                    Bu_(:,:,k) = x__;

                    % Build two small “surfaces” by repeated solves under
                    % reconstructed and true objectives on random probes,
                    % then compute a Hausdorff-like max-min set distance.
                    measmin = 10; %#ok<NASGU> % (unused)
                    xmin    = zeros(2,M); %#ok<NASGU> % (unused)
                    testnum = 10;
                    s1      = zeros(2,M,testnum); % reconstructed surface samples
                    s2      = s1;                 % true surface samples

                    for test=1:testnum
                        alph = [rand+0.1, rand+0.1];

                        x0   = abs(randn(2,M));
                        [x__, ~] = fmincon(fcn2, x0, A__, b__, Aeq, beq, lb, ub, @(x) RPcon2(x,alph,y_,M), options);
                        s1(:,:,test) = x__;

                        x0   = abs(randn(2,M));
                        [x_, ~] = fmincon(fcn,  x0, A__, b__, Aeq, beq, lb, ub, @(x) RPcon2(x,alph,y_,M), options);
                        s2(:,:,test) = x_;
                    end

                    % Compute symmetric max-min distance between the two sets {s1} and {s2}
                    dmin1 = 100*ones(1,testnum);
                    dmin2 = dmin1;
                    for j=1:testnum
                        for l=1:testnum
                            dist1 = sum(sum(abs(s1(:,:,j) - s2(:,:,l))));
                            dist2 = sum(sum(abs(s2(:,:,j) - s1(:,:,l))));
                            if (dist1 < dmin1(j)), dmin1(j) = dist1; end
                            if (dist2 < dmin2(j)), dmin2(j) = dist2; end
                        end
                    end
                    HD(k) = max(max(dmin1,dmin2)); % Hausdorff-like discrepancy
                end

                % Average the per-probe distance for this run
                dift(mc_out,mc_in,n,rg) = sum(HD)/T;
            end
        end
        fprintf('mc_in: %i/%i, mc_out: %i/%i\n',mc_in,MC_in,mc_out,MC_out);
    end
end

time=toc; % total runtime

%% Data-Processing:
% Compute statistics that compare classical vs robust reconstruction.
% diftd1: distances for "naive" (classical Afriat) under noise
% diftd2: distances for "robust" under noise
diftd1 = zeros(MC_out,MC_in,length(rg_space));
diftd2 = diftd1;

% Per-outer max over inner runs; then average across outer runs:
diftd1_max     = zeros(MC_out,length(rg_space));
diftd1_max_avg = zeros(1,length(rg_space));
diftd1_std     = diftd1_max_avg;

diftd2_max     = diftd1_max;
diftd2_max_avg = zeros(1,length(rg_space));
diftd2_std     = diftd1_std;

% Average of the LP/robust metrics over Monte Carlo
met_avg = zeros(length(nspace),length(rg_space));

for rg = 1:length(rg_space)
    for n = 1:length(nspace)
        met_avg(n,rg) = sum(sum(met(:,:,n,rg)))/(MC_out*MC_in);
    end

    for mc_out = 1:MC_out
        for mc_in=1:MC_in
            % n=2 -> classical with noise; n=3 -> robust with noise
            diftd1(mc_out,mc_in,rg) = dift(mc_out,mc_in,2,rg);
            diftd2(mc_out,mc_in,rg) = dift(mc_out,mc_in,3,rg);
        end
        % worst case across inner runs
        diftd1_max(mc_out,rg) = max(diftd1(mc_out,:,rg));
        diftd2_max(mc_out,rg) = max(diftd2(mc_out,:,rg));
    end
    % average worst-case across outer runs
    diftd1_max_avg(rg) = sum(diftd1_max(:,rg))/MC_out;
    diftd2_max_avg(rg) = sum(diftd2_max(:,rg))/MC_out;

    % standard deviation across outer runs (for error bars)
    diftd1_std = std(diftd1_max);
    diftd2_std = std(diftd2_max);
end

% Plot average of worst-case (max) distances vs noise level
figure;
errorbar(rg_space,diftd1_max_avg, diftd1_std.^2,'r'); hold on % naive (classical Afriat)
errorbar(rg_space,diftd2_max_avg, diftd2_std.^2,'b');        % robust
title('avg max'); legend('naive','robust');

% Print summary stats
fprintf('naive max avg: %0.4f\n',diftd1_max_avg)
fprintf('naive max std: %0.4f\n', diftd1_std)
fprintf('robust max avg: %0.4f\n',diftd2_max_avg)
fprintf('robust max std: %0.4f\n',diftd2_std)

% Mean distances across all runs (not the per-outer max)
avgs1 = sum(sum(diftd1))/(MC_out*MC_in);
avgs2 = sum(sum(diftd2))/(MC_out*MC_in);

% Std across inner (flattened) for plotting error bars
stds1 = std(reshape(diftd1,1,[],length(rg_space)));
stds2 = std(reshape(diftd2,1,[],length(rg_space)));

% Extract vectors for each noise setting
avgs1_ = zeros(1,length(rg_space));
avgs2_ = avgs1_;
stds1_ = avgs1_;
stds2_ = avgs1_;
for i=1:length(rg_space)
    avgs1_(i) = avgs1(:,:,i);
    avgs2_(i) = avgs2(:,:,i);
    stds1_(i) = stds1(:,:,i);
    stds2_(i) = stds2(:,:,i);
end

% Plot mean distances vs noise level
figure;
errorbar(rg_space,avgs1_,stds1_.^2,'r'); hold on
errorbar(rg_space,avgs2_,stds2_.^2,'b');
title('avg'); legend('naive','robust');

% Print mean/std summaries
fprintf('\n naive avg: %0.4f\n', avgs1_)
fprintf('naive std: %0.4f\n', stds1_)
fprintf('robust avg: %0.4f\n',avgs2_)
fprintf('robust std: %0.4f\n',stds2_)

% post-process CV_st data (robust constraint violations per iteration):
% Aggregate across MC runs to get an average violation curve per iteration.
pl_   = sum(sum(CV_st))/(MC_out*MC_in);
std__ = sum(std(CV_st));
pl    = zeros(100,length(rg_space));
std_  = pl;
for i=1:100
    for k=1:length(rg_space)
        pl(i,k)   = pl_(:,:,i,k);
        std_(i,k) = std__(:,:,i,k);
        if (pl(i,k) < 0)
            pl(i,k) = 0;
        end
    end
end
cols = ['g','b','r','k','c','m','k']; %#ok<NASGU> % (reserved for multi-curve plotting)

% =========================
% Helpers (unchanged form)
% =========================
% af_eval: evaluates the maximum Afriat inequality violation given (u,lambda)
%          and a set of probe/response samples (A,B). A lower value is better.
function [sol] = af_eval(u,lambda,A,B,N,M)
    LParray = zeros(N,N,M);
    for s=1:N
        for t=1:N
            for i=1:M
                LParray(s,t,i) = (u(s,i) - u(t,i))/lambda(t,i) - ...
                         A(:,t)'*(B(:,i,s) - B(:,i,t));
            end
        end
    end
    sol = max(max(max(LParray)));
end

% mcval: objective for the “max constraint violation” subproblem in the robust loop.
%        It returns val1 + val2, where:
%        - val1 is the max Afriat violation over (s,t,i) for candidate B
%        - val2 penalizes the L2 deviation of B from the noisy observations b_sol_
%          with coefficients v(2) and v(1) (regularization/offset).
function [sol] = mcval(u,lambda,v,A,b_sol_,B,N,M)
    LParray = zeros(N,N,M);
    for s=1:N
        for t=1:N
            for i=1:M
                LParray(s,t,i) = (u(s,i) - u(t,i))/lambda(t,i) - ...
                         A(:,t)'*(B(:,i,s) - B(:,i,t));
            end
        end
    end
    val1 = max(max(max(LParray)));

    L2sum = 0;
    for i=1:M
        for t=1:N
              L2sum = L2sum + sqrt((B(1,i,t) - b_sol_(1,i,t))^2 + ...
                  (B(2,i,t) - b_sol_(2,i,t))^2);
        end
    end
    val2 = -v(2)*L2sum - v(1);

    sol = val1 + val2; % maximize
end

% mcval2: used in the finite-reduction step; returns the maximum over past
%         adversarial samples (stored in B) of the same val1 + val2 structure,
%         i.e., the worst-case loss across the B_set accumulated so far.
function [sol] = mcval2(u,lambda,v,A,b_sol_,B,N,M,itr)
    vmax = -10;
    for it=1:itr-1
        LParray = zeros(N,N,M);
        for s=1:N
            for t=1:N
                for i=1:M
                    LParray(s,t,i) = (u(s,i) - u(t,i))/lambda(t,i) - ...
                             A(:,t)'*(B(:,i,s,it) - B(:,i,t,it));
                end
            end
        end
        val1 = max(max(max(LParray)));

        L2sum = 0;
        for i=1:M
            for t=1:N
                  L2sum = L2sum + sqrt((B(1,i,t,it) - b_sol_(1,i,t))^2 + ...
                      (B(2,i,t,it) - b_sol_(2,i,t))^2);
            end
        end
        val2 = -v(2)*L2sum; % - v(1);

        sum_ = val1 + val2; %#ok<NASGU>
        if (sum_ >= vmax)
            vmax = sum_;
        end
    end
    sol = vmax;
end

% L1dist: (auxiliary) computes an L1 discrepancy between reconstructed U and
%         a set of hard-coded “true” forms on a grid (not used in plots).
function [dist] = L1dist(U,sc,betaspace)
% L1 distance btw true and reconstructed utilities:
    sum_ = zeros(1,3);
    for b1 = 1:length(betaspace)
        for b2 = length(betaspace)
            sum_(1) = sum_(1) + abs(sc(1)*U(b1,b2,1) - betaspace(b1) + betaspace(b2));
            sum_(2) = sum_(2) + abs(sc(2)*U(b1,b2,2) - betaspace(b1) + betaspace(b2)^(1/4));
            sum_(3) = sum_(3) + abs(sc(3)*U(b1,b2,3) - betaspace(b1)^(1/4) + betaspace(b2));
        end
    end
    dist = sum(sum_);
end

% RPcon3: vectorized inequality A'B <= y for each probe k. Used inside robust
%         subproblems to constrain the adversarial B_var samples.
function [c] = RPcon3(x,a,y,M,N)
% A'B <= y (per k)
c = zeros(N,1);
for k=1:N
    c(k) = sum(a(:,k)'*x(:,:,k)) - y(k);
end
end
