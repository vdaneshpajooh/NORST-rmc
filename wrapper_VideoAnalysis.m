%%%Wrapper to call the matrix completion function, perform the task of Subspace
%%%Tracking with missing data

clear;
clc;
% close all
load('/home/vahidd/Git/Research/NORST-random/data/Han_Data/Curtain.mat')

addpath('YALL1_v1.4')
addpath('PROPACK')
addpath('PG-RMC')
addpath('/home/vahidd/Git/Research/NORST-robust/PG-RMC/Mex')
addpath('RPCA-GD')


% Lx = M;
% [~,m] = size(Lx);
% L = zeros(prod(imSize/2),m);
% for i = 1:m
%     x = reshape(Lx(:,i), imSize);
%     L(:,i) = reshape(x(1:2:end,1:2:end),[prod(imSize/2),1]);
% end
% 
% [~,m] = size(DataTrain);
% Train = zeros(prod(imSize/2),m);
% for i = 1:m
%     x = reshape(DataTrain(:,i), imSize);
%     Train(:,i) = reshape(x(1:2:end,1:2:end),[prod(imSize/2),1]);
% end

L = I;
Train = DataTrain;
%% Parameter Initialization
[n,m] = size(L);

r = 30;

t_max = m;

alpha = 60;

%%% TOLERANCE %%%
% tolerance used in cgls(conjugate gradient least squares)
tol = 1e-10;

t_back = t_max;

GRASTA = 1;
norst = 1;
NCRMC = 1;
RPCAGD = 1;

%% Generating missing entries

%     rho = 0.1; %denotes fraction of missing entries
%     BernMat = rand(n, t_max);
%     T = 1 .* (BernMat <= 1 - rho);
    
%     (Moving Square)
    T = ones(n,m);
    
    height = imSize(1);
    width = imSize(2);
    
    a = 28;
    b = 32;
    
    idx_frame = [width * 0 + 1 : width * 15];
    smin = 0;
    smax = 1;
    for j = idx_frame   
        for i = smin:smax
            T(height*i+ a : height*i + b ,j) = zeros(b-a+1,1);
        end
        smax = smax+1;
        if(smax - smin > width/4)
            smin = smin + 1;
        end
        
        if(smax >= width)
            smax = 1;
            smin = 0;
        end
        
    end

M = L .* T;
    
%% Calling the Algorithms

if(norst == 1)
        %%% NORST-random %%%
    % Algorithm parameters for NORST
    lambda = 0;
    K = 3;
    ev_thresh = 2e-3;
    omega = 15 ;
    mu = mean(Train,2);
%     mu = zeros(1,t_max);
    M_norst = M - mu;

    fprintf('\tNORST\n')
%     Initialization of true subspace
    t_norst = tic;
    fprintf('Initialization...\t');
        P_init = orth(ncrpca(Train, r, 1e-2, 100));        
%         Train = DataTrain - mu;
%         [P_init, ~] = svds(Train,r);
        
    fprintf('Subspace initialized\n'); 
    fprintf('iteration:\n');
    [x_cs_hat, FG, BG, L_hat, P_hat, S_hat, T_hat, t_hat, ...
            P_track_full, t_calc] = ...
            NORST_video(M_norst, mu, T,...
            P_init, ev_thresh, alpha, K, omega);
    t_NORST = toc(t_norst);                
    
%     err_L_fro_norst = norm(L-BG,'fro')/norm(L,'fro');
%     err_nmse_norst = sqrt(mean((L - BG).^2, 1)) ./ sqrt(mean(L.^2, 1));
end

  if(GRASTA == 1)
        fprintf('GRASTA\t');
        
%         [I,J] = find(T);
        t_grasta = tic;
        run_alg_grasta
%        L_hat_grasta = Usg * Vsg';
       t_GRASTA = toc(t_grasta);
       
%        err_L_fro_GRASTA = norm(L-L_hat_grasta,'fro')/norm(L,'fro');
%        err_nmse_grasta = sqrt(mean((L - L_hat_grasta).^2, 1)) ./ sqrt(mean(L.^2, 1));
  end
  
   if(NCRMC == 1)
       t_ncrmc = tic;
%        avg = mean(mean(M,1),2);
%        M2 = M - avg;
       fprintf('NC_RMC\n');
       
       [U_t, SV_t] = ncrmc(M,T);
       L_hat_ncrmc = U_t * SV_t;
       t_NCRMC = toc(t_ncrmc);
       
%        err_L_fro_ncrmc = norm(L-L_hat_ncrmc,'fro')/norm(L,'fro');
%        err_nmse_ncrmc = sqrt(mean((L - L_hat_ncrmc).^2, 1)) ./ sqrt(mean(L.^2, 1));
   end
   
   if(RPCAGD == 1)
       
       r = 30;
       alpha = 0.1;
        % Decomposition via Gradient Descent
        % algorithm paramters
        params.step_const = 0.5; % step size parameter for gradient descent
        params.max_iter   = 30;  % max number of iterations
        params.tol        = 2e-4;% stop when ||Y-UV'-S||_F/||Y||_F < tol
        % alpha_bnd is some safe upper bound on alpha, 
        % that is, the fraction of nonzeros in each row of S (can be tuned)
        gamma = 1.5;
        alpha_bnd = gamma*alpha;
        
        
       fprintf('RPCA-GD\n');
       t_rpcagd = tic;
       [U,V] = rpca_gd(M, r, alpha_bnd, params);
       L_hat_rpcagd = U*V'; % low-rank
       S_hat_rpcagd = M - L_hat_rpcagd; % sparse
       
       t_RPCAGD = toc(t_rpcagd);
       
%        err_L_fro_ncrmc = norm(L-L_hat_ncrmc,'fro')/norm(L,'fro');
%        err_nmse_ncrmc = sqrt(mean((L - L_hat_ncrmc).^2, 1)) ./ sqrt(mean(L.^2, 1));
   end

%% Display the reconstructed video
% save('video_curtain_otherAlgos_rho30.mat')
% DisplayVideo(L, T, M, BG, imSize,'Curtain_movobj_fgbg.avi')