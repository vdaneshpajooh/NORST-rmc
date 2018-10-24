function [x_cs_hat, L_hat, P_hat, S_hat, T_hat, t_hat, ...
    P_track_full, T_calc]= NORST(M,T_miss, ...
    P_init, ev_thresh, alpha, K, omega)
%This folder contains the code accompanying pre-print.
%[1] "Nearly Optimal Robust Subspace Tracking", Praneeth Narayanamurthy and Namrata Vaswani, arXiv:1712.06061, 2017.

%%%                          Inputs                         %%%
%%%     M - measurement matrix                              %%%
%%%     ev_thres - threshold for subspace change detection  %%%
%%%     P_init - an initial estimate of the subspace        %%%
%%%     t_train - the dimension of the training data        %%%


%%%                       Algorithm parameters              %%%
%%%     alpha - frame length                                %%%
%%%     K - number of projection PCA steps                  %%%
%%%     omega - threshold for non-zero value in S           %%%
%%% 	K_CS - number of CoSaMP iterations 		    %%%
%%% 	outc - an upper bound on estimate of fraction of outliers per column

%%%                          Outputs                        %%%
%%%     L_hat - estimate of the low rank matrix             %%%
%%%     P_hat - estimate of the subspace in which L lies    %%%
%%%     S_hat - estimate of the sparse signal               %%%
%%%     t_hat - estimate of subspace change times           %%%

%% Initializations
%thresh = ev_thresh / 2;
[~, r_init] = size(P_init);
% t_train = 500;
% [P_init, ~] = svds(M_t(:,1:t_train),r_init);

% M = M_t(:,t_train+1 : end);
[n, t_max] = size(M);
% T_miss = T(:,t_train+1:end);

P_hat_old = P_init;
P_hat_new = [];
P_hat = [P_hat_old, P_hat_new];

%% Mean Subtraction (just for video case)
% M_mean = mean(M,1);
% M = M - M_mean;

T_hat = zeros(n, t_max);
S_hat = zeros(n, t_max);

L_hat = zeros(n, t_max);
BG = zeros(n,t_max);
% FG = zeros(n, t_max);

t_hat = [];

% L_hat(:, 1 : t_train) = M(:, 1 : t_train);
k = 0;
cnt = 1;
phi_t = (eye(n) - P_hat * P_hat');
% In = eye(n);
ph = 1;     %ph - 0 => detect, 1 => ppca

opts.delta=0.4;

%% Main Algorithm Loop
for ii = 1 : t_max
    %% Estimate support
    Atf.times = @(x) x - (P_hat * (P_hat' * x));
    Atf.trans = @(y) y - (P_hat * (P_hat' * y));
    phi.times = @(x) x - (P_hat_old * (P_hat_old' * x));
    y_t = Atf.times(M(:, ii));
    opts.tol   = 1e-4;
    opts.print = 0;
    
    weights = ones(n,1); 
    lambda1 = 0;
    weights(find(T_miss(:,ii) == 0)) = lambda1;
    opts.weights = weights;
    
%%% used for synthetic data
    opts.delta = omega * 2 / 15;

%%% videos
%     opts.delta = norm(Atf.times(L_hat(:,ii-1)));
% 
%     opts.tol = 1e-3;
%     opts.print = 0;
    
    x_t_hat_cs = yall1(Atf, y_t, opts);
%     omega = sqrt(M(:, ii)' * M(:, ii) / n);
    
    x_cs_hat(:,ii) = x_t_hat_cs;
    
    t_hat_temp = find(abs(x_t_hat_cs) > omega);
    T_hat(t_hat_temp, ii) = 255;
    
%     S_hat_sparse(t_hat_temp,ii) = phi_t(:,t_hat_temp)\y_t;
%     y_tt = Atf.times(M(:, ii) - S_hat_sparse(:,ii));

    
    %LS.times = @(x) phi(:, t_hat_temp) * x;
    %LS.trans = @(y) phi(:, t_hat_temp)' * x;
    
    %y_t = M(:, ii) - (P_hat * (P_hat' * M(:, ii)));
    %DecayRate = 0.9; %other values work, may make it slower
    %x_t_hat_cs = cosamp_cgls(phi_t, ...
    %    y_t, outc, DecayRate, K_CS, 1e-6);
    %t_hat_temp = find(abs(x_t_hat_cs) > omega);
    %     T_hat(t_hat_temp, ii) = 1;
    
    %% Estimate signal components
    % %         [S_hat(t_hat_temp, ii), ~] = ...
    % %             lsqr(phi_t(:, t_hat_temp), y_t, 1e-6, 50);
    %     S_hat(t_hat_temp, ii) = phi_t(:, t_hat_temp) \ y_t;

        T_union = unique([t_hat_temp;find(T_miss(:,ii) == 0)]);
%         T_union = find(T_miss(:,ii) == 0);
%     T_union = t_hat_temp;
    %T_hat(T_union, ii) = 255;
    S_hat(T_union, ii) = cgls(phi_t(:, T_union), y_t, ...
        0, 1e-16, 20);
%     S_hat(T_union,ii) = phi_t(:,T_union) \ y_t;
    L_hat(:, ii) = M(:, ii) - S_hat(:, ii);
    
    %just for video case
%     FG(T_union, ii) = M(T_union, ii);
%     BG(:,ii) = L_hat(:,ii) + M_mean(ii);
    
    %% Subspace update
    if(~mod(ii + 1 , alpha))
        u = (ii + 1) / alpha;
        idx = (u-1) * alpha + 1 : u * alpha ;
        L_temp = L_hat(:, idx);
        
        %MM = L_temp - (P_hat_old *(P_hat_old' * L_temp));
        MM = phi.times(L_temp);
        
        if(~ph)     %%detect phase
            %             phi_t = eye(n) - P_hat * P_hat';
            if(svds(MM, 1) >= sqrt(alpha * ev_thresh))
                ph = 1;
                t_hat = [t_hat, ii];
                k = 0;
            end
        else        %%pca phase
            P_hat = simpleEVD((L_hat(:, max(1, ii - alpha + 1) : ii)), r_init);
%             P_hat = BlockStochPowerMethodGenRank((L_hat(:, max(1, ii - alpha + 1) : ii))...
%                 , r_init, block_size);
            phi_t = speye(n) - P_hat * P_hat';
            k = k + 1;
            
            if(k==K + 1)
                P_hat_old = P_hat;
                k = 1;
                ph = 0;
                phi_t = speye(n) - P_hat * P_hat';
            end
        end
    %end
    
    
    %% Return subspace
%    if((ii == 1) || ~(mod(ii + 1, alpha)))
        P_track_full{cnt} = P_hat;
        %P_track_new{cnt} = P_hat_new;
        T_calc(cnt) = ii;
        cnt = cnt + 1;
    end
end
end
