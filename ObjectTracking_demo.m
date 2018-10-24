%%%Wrapper to call the matrix completion function, perform the task of Subspace
%%%Tracking with missing data

clear;
clc;
% close all
load('/home/vahidd/Git/Research/NORST-robust/highway/highway.mat')

addpath('YALL1_v1.4')
addpath('PROPACK')

for i = 1:numel(Im)
    img_ds_2x{i} = double(img_gray{i}(1:4:end,1:4:end));
end

imSize = size(img_ds_2x{1});

for i = 1:numel(img_gray)
    L(:,i) = reshape(img_ds_2x{i},prod(imSize),1);
end

%% Parameter Initialization
[n,m] = size(L);

r = 5;

t_max = m;

alpha = 10;

%%% TOLERANCE %%%
% tolerance used in cgls(conjugate gradient least squares)
tol = 1e-3;

t_back = t_max;

GRASTA = 0;
norst = 1;
NCRMC = 0;

t_train = 10;
M = L(:,t_train+1:end);
T = ones(size(M));

[~,t_max] = size(M);
%% Calling the Algorithms

%%% NORST-random %%%
% Algorithm parameters for NORST
lambda = 0;
K = 3;
ev_thresh = 2e-3;
omega = 25;

mu = mean(M,2);
% mu = zeros(1,t_max);
M_norst = M - mu;

if(norst == 1)
    fprintf('\tNORST\n')
%     Initialization of true subspace
    t_norst = tic;
    fprintf('Initialization...\t');
%         P_init = orth(ncrpca(DataTrain, r, 1e-2, 200));        
        [P_init, ~] = svds(L(:,1:t_train),r);
        
    fprintf('Subspace initialized\n'); 
    
    [x_cs_hat, FG, BG, L_hat, P_hat, S_hat, T_hat, t_hat, ...
            P_track_full, t_calc] = ...
            NORST_video(M_norst, mu, T, P_init, ev_thresh, alpha, K, omega);
    t_NORST = toc(t_norst);                
end


%% Compute Performance Metrics
%compute the "normalized-mse (nmse) && frobenius norm"

% if (norst == 1)
%     err_L_fro_norst = norm(L - L_hat,'fro')/norm(L,'fro');
%     err_nmse_norst = sqrt(mean((L - L_hat).^2, 1)) ./ sqrt(mean(L.^2, 1));
% end

%% Display the reconstructed video
% save('video_curtain_otherAlgos_rho30.mat')
DisplayVideo(L(:,1+t_train:end), T, M_norst, S_hat, imSize,'highway.avi')