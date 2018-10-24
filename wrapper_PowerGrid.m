%%%Wrapper to call the matrix completion function, perform the task of Subspace
%%%Tracking with missing data

clear;
clc;
% close all
load('Gen1Vref_0.38006Hz_res.mat')

addpath('YALL1_v1.4')
addpath('PROPACK')


L = v_mag(602:end,:);
L = 1e5 * L';

%% Parameter Initialization
[n,m] = size(L);



r = 30;

t_max = m;

alpha = 60;

%%% TOLERANCE %%%
% tolerance used in cgls(conjugate gradient least squares)
tol = 1e-10;

t_train = 400;

T = ones(n,t_max - t_train);
M = L(:,t_train + 1 : end);
%% Calling the Algorithms
        %%% NORST-random %%%
    % Algorithm parameters for NORST
    lambda = 0;
    K = 5;
    ev_thresh = 2e-4;
    omega = 0.0005 ;
    mu = mean(L,2);

    fprintf('\tNORST\n')
%     Initialization of true subspace
    t_norst = tic;
    fprintf('Initialization...\t');
%         P_init = orth(ncrpca(L(:,1:t_train), r, 1e-3, 100));        
        [P_init, ~] = svds(L(:,1:t_train),r);
        
    fprintf('Subspace initialized\n'); 
    
    [x_cs_hat, FG, BG, L_hat, P_hat, S_hat, T_hat, t_hat, ...
            P_track_full, t_calc] = ...
            NORST_video(M, mu, T,...
            P_init, ev_thresh, alpha, K, omega);
    t_NORST = toc(t_norst)
    
    figure
    imagesc(x_cs_hat)
    title('outliers')
    
    figure
    imagesc(L)
    title('measurements')