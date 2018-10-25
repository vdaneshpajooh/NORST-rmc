clear;
clc;
% close all

addpath('YALL1_v1.4')
addpath('PROPACK')
% addpath('PG-RMC')
% addpath('/home/vahidd/Git/Research/NORST-robust/PG-RMC/Mex')
% addpath('GRASTA')
addpath('RPCA-GD')
%% Algorithms to run
NORSTR = 0;
GRASTA = 0;
NCRMC = 0;
RPCAGD = 1;
%% Data Generation
n = 1000;
t_max = 10000;
s = 100;

miss_s = 0;
alpha = 60;
alpha1 = 100;
f = 100;
cnt = 1;
MC = 1;

t_calc_pca = alpha-1:alpha:t_max;

%NORST
temp_SE_NORST = zeros(length(t_calc_pca), MC);
temp_err_L_NORST = zeros(t_max, MC);
t_NORST = 0;
err_L_fro_NORST = zeros(MC,1);

%RPCAGD
temp_SE_RPCAGD = zeros(length(t_calc_pca), MC);
temp_err_L_RPCAGD = zeros(t_max, MC);
t_RPCAGD = 0;
err_L_fro_RPCAGD = zeros(MC,1);

%GRASTA
temp_SE_GRASTA = zeros(length(t_calc_pca), MC);
temp_err_L_GRASTA = zeros(t_max, MC);
t_GRASTA = 0;
err_L_fro_GRASTA = zeros(MC,1);

%NC-RMC
temp_SE_NORST = zeros(length(t_calc_pca), MC);
temp_err_L_NORST = zeros(t_max, MC);
t_NORST = 0;
err_L_fro_NORST = zeros(MC,1);

% err_t = zeros(MC, 16);
start_time = clock;
for mc = 1 : MC
            
            fprintf('Monte-Carlo iteration %d in progress\n', mc);
            
            %%%Generating support set and sparse vectors
            t_train = 0;
            S = zeros(n, t_max);
            rho_s = 1;
            b0 = 0.1;
            beta = ceil(b0 * alpha1);
            x_max = 25;
            x_min = 15;
            alpha1 = 100;
            num_changes = floor((t_max -t_train)/beta);
            
            num_changes1 = min(floor(alpha1 / beta), ceil(n/s));
            
            flag = 0;
            ii1 = 1;
            fval1 = 0;
            for ii = 1 : num_changes
                if(~flag)   %%downward motion
                    if(ii1 <= num_changes1)
                        bind = fval1 + (ii1 - 1) * s/rho_s + 1;
                        sind = min(bind - 1 + s, n);
                        ii1 = ii1 + 1;
                        if(ii1 == num_changes1 + 1)
                            flag = 1;
                            ii1 = 1;
                            fval2 = bind;
                        end
                    end
                else
                    if(ii1 <= num_changes1)
                        bind = max(fval2 - (ii1 - 1) * s/rho_s , 1);
                        sind = bind - 1 + s;
                        ii1 = ii1 + 1;
                        if(ii1 == num_changes1 + 1)
                            flag = 0;
                            ii1 = 1;
                        end
                    end
                end
                idx = bind : sind;
                jdx = t_train + (ii-1) * beta + 1 : t_train + ii * beta;
                S(idx, jdx) = x_min + ...
                    (x_max - x_min) * rand(length(idx), beta);
                T(idx, jdx) = 1;
            end
            
            fprintf('fraction of sparse entries: %d \n',length(find(T(:) == 1)) / numel(T));
            
            t_train = 400;
    %%% bernoulli model for outliers
        rho = 0.1;%Rho(i); % 1-rho : denotes number of outliers
        BernMat = rand(n, t_max);
        temp_miss = 1 .* (BernMat <= 1 - rho);
        T_miss = temp_miss;
        %imagesc(T)
        

%         S_temp = x_min + (x_max - x_min) * rand(n,t_max);
%         S = S_temp .* (1-T);
    
    %%%Generating low-rank matrix
        r_0 = 30;
        r_1 = 0;
        r_2 = 0;
        r = r_0;
        L = zeros(n, t_max);
        
        diag_entries = [linspace(sqrt(f), sqrt(f)/2, r_0)];
        t_1 = 4000;
        t_2 = 8000;
        coeff_train = zeros(r_0, t_max);
        
        for cc = 1 : r_0
            coeff_train(cc, :) = -diag_entries(cc) + ...
                2 * diag_entries(cc) * rand(1, t_max);
        end
        
        Btemp1 = randn(n);
        B1 = (Btemp1 - Btemp1')/2;
        Btemp2 = randn(n);
        B2 = (Btemp2 - Btemp2')/2;
        
        delta1 = 0.5e-3;
        delta2 = 0.8 * delta1;
        
        P = orth(randn(n, r_0));
        PP1 = expm(delta1 * B1)  * P;
        PP2 = expm(delta2 * B2) * PP1;
        
        L(:, 1:t_1) = P(:, 1:r_0) * coeff_train(:, 1:t_1);
        L(:, t_1+1:t_2) = PP1 * coeff_train(:, t_1+1:t_2);
        L(:, t_2 + 1 : end) = PP2 * coeff_train(:, t_2+1:end);
%         T_miss(:,1:t_train) = ones(n,t_train);
        M = L + S;
        M = M .* T_miss;
        M_rpca = M(:,t_train+1:end);
%         M_rpca = M(:,t_train+1:end).* T_miss;% .* T_miss ;
%         M_rpca(M_rpca==0) = 10 * x_max;
%         M_mc = L .* T_miss;
        
        %% Call to GRASTA
        if(GRASTA == 1)
        fprintf('GRASTA\t');
        maxCycles = 1;
        
        OPTIONS.QUIET               = 1;     % suppress the debug information

        OPTIONS.MAX_LEVEL           = 20;    % For multi-level step-size,
        OPTIONS.MAX_MU              = 15;    % For multi-level step-size
        OPTIONS.MIN_MU              = 1;     % For multi-level step-size

        OPTIONS.DIM_M               = n;  % your data's ambient dimension
        OPTIONS.RANK                = r; % give your estimated rank

        OPTIONS.ITER_MIN            = 20;    % the min iteration allowed for ADMM at the beginning
        OPTIONS.ITER_MAX            = 20;    % the max iteration allowed for ADMM
        OPTIONS.rho                 = 2;   % ADMM penalty parameter for acclerated convergence
        OPTIONS.TOL                 = 1e-8;   % ADMM convergence tolerance

        OPTIONS.USE_MEX             = 0;     % If you do not have the mex-version of Alg 2
                                         % please set Use_mex = 0.
        
        CONVERGE_LEVEL = 20;
        
        [I,J] = find(T_miss(:,1+t_train:end));
        t_grasta = tic;
       [Usg, Vsg, Osg] = grasta_mc(I,J,M_rpca(find(T_miss(:,1+t_train:end))),n,t_max-t_train,maxCycles,CONVERGE_LEVEL,OPTIONS); 
       L_hat_grasta = Usg * Vsg';
       t_GRASTA = toc(t_grasta)
       err_L_fro_GRASTA(mc) = norm(L(:,t_train + 1:end)-L_hat_grasta,'fro')/norm(L(:,t_train+1:end),'fro');
        end
       
           %% Call to NO-RMC
       if(NCRMC == 1)
       t_ncrmc = tic;
           avg = mean(mean(M_rpca,1),2);
       M2 = M_rpca - avg;
       fprintf('NO_RMC\t');
       
       [U_t, SV_t] = ncrmc(M2,T_miss(:,1+t_train:end));
       L_hat_ncrmc = U_t * SV_t + avg;
       t_NCRMC = toc(t_ncrmc)
       err_L_fro_ncrmc(mc) = norm(L(:,t_train+1:end)-L_hat_ncrmc,'fro')/norm(L(:,t_train+1:end),'fro');
       end
       
        %% Call to pROST
       if(RPCAGD == 1)
       t_rpcagd = tic;
           
       fprintf('RPCAGD\t');
        alpha = 0.1;  % sparsity of S* (expected)
        % Decomposition via Gradient Descent
        % algorithm paramters
        params.step_const = 0.5; % step size parameter for gradient descent
        params.max_iter   = 30;  % max number of iterations
        params.tol        = 2e-4;% stop when ||Y-UV'-S||_F/||Y||_F < tol
        % alpha_bnd is some safe upper bound on alpha, 
        % that is, the fraction of nonzeros in each row of S (can be tuned)
        gamma = 1.5;
        alpha_bnd = gamma*alpha;

        [U,V] = rpca_gd(M_rpca, r, alpha_bnd, params);
        L_hat_RPCAGD = U*V'; % low-rank
        S_hat_RPCAGD = M_rpca - L_hat_RPCAGD; % sparse
       t_RPCAGD = toc(t_rpcagd)
       
       err_L_fro_RPCAGD(mc) = norm(L(:,t_train+1:end)-L_hat_RPCAGD,'fro')/norm(L(:,t_train+1:end),'fro');
       end
       
        %% Calls to NORST
        if(NORSTR == 1)
        %%%Algorithm parameters
        lambda = 0;
        K = 36;
        omega = x_min / 2;
        %     gamma = sqrt(4 * log(n)/n);
        %     s = ceil((gamma + rho) * n);
        
        %%%Call to NORST
        fprintf('NORST\t');
        ev_thresh = 7.5961e-04;
        t_norst = tic;
        M_train = M(:,1:t_train);
        M_train(M_train == 0)= x_max;
        P_init = orth(ncrpca(M_train, r, 1e-3, 100));
%         [P_init,~] = svds(M_train,r);
%         sigma = 1e-6;
%         P_init = orth(P + sigma * randn(n,r));
% %         P_init = orth(randn(n,r));
% %         P_init = zeros(n,r);
%         Calc_SubspaceError(P,P_init)
        [x_cs_hat, L_hat_rpca, P_hat_rpca, S_hat_rpca, T_hat_rpca, t_hat_rpca, ...
            P_track_full_rpca, t_calc_rpca] = ...
            NORST(M_rpca(:,  1 :end),...
            T_miss(:, 1+t_train : end), P_init, ev_thresh, alpha, K, omega);
        t_NORST = toc(t_norst)
       err_L_fro_NORST(mc) = norm(L(:,t_train+1:end)-L_hat_rpca,'fro')/norm(L(:,t_train+1:end),'fro');
        end
%% Compute performance metrics
        if(NORSTR == 1)
        %%% norst-rmc
        tempRPCA_err_L(mc, :) = ...
            sqrt(mean((L(:, t_train + 1 : end) - L_hat_rpca).^2, 1)) ./ ...
            sqrt(mean(L(:, t_train + 1 : end).^2, 1));
        
        miss_s = ...
            miss_s + (length(find(S_hat_rpca))- length(find(S)))/numel(S);
        end
        if(NCRMC == 1)
        %%% ncrmc
        tempNCRMC_err_L(mc, :) = ...
            sqrt(mean((L(:, t_train + 1 : end) - L_hat_ncrmc).^2, 1)) ./ ...
            sqrt(mean(L(:, t_train + 1 : end).^2, 1));
        end
        
        if(RPCAGD == 1)
        %%% prost
        tempRPCAGD_err_L(mc, :) = ...
            sqrt(mean((L(:, t_train + 1 : end) - L_hat_RPCAGD).^2, 1)) ./ ...
            sqrt(mean(L(:, t_train + 1 : end).^2, 1));
        end
        
        if(GRASTA == 1)
        %%% grasta 
        tempGRASTA_err_L(mc, :) = ...
            sqrt(mean((L(:, t_train + 1 : end) - L_hat_grasta).^2, 1)) ./ ...
            sqrt(mean(L(:, t_train + 1 : end).^2, 1));    
        end
        %%Calculate the subspace error
        for jj = 1 : length(t_calc_pca)
            if(t_calc_pca(jj) < t_1)
                if(NORSTR == 1)
                tempRPCA_SE_Phat_P(mc, jj) = ...
                Calc_SubspaceError(P_track_full_rpca{jj}, P);
                end
                if(GRASTA == 1)
                P_track_GRASTA = orth(L_hat_grasta(:, 1 : t_1 - t_train));
                tempGRASTA_SE_Phat_P(mc, jj) = ...
                Calc_SubspaceError(P_track_GRASTA, P);   
                end
                if(NCRMC == 1)
                P_track_NCRMC = orth(L_hat_ncrmc(:, 1 : t_1 - t_train));
                tempNCRMC_SE_Phat_P(mc, jj) = ...
                Calc_SubspaceError(P_track_NCRMC, P);   
                end
                
                if(RPCAGD == 1)
                P_track_RPCAGD = orth(L_hat_RPCAGD(:, 1 : t_1 - t_train));
                tempRPCAGD_SE_Phat_P(mc, jj) = ...
                Calc_SubspaceError(P_track_RPCAGD, P);   
                end
                
            elseif((t_calc_pca(jj) >= t_1) && (t_calc_pca(jj) < t_2))
                if(NORSTR == 1)
                tempRPCA_SE_Phat_P(mc, jj) = ...
                Calc_SubspaceError(P_track_full_rpca{jj}, PP1);
                end
                
                if(GRASTA == 1)
                P_track_GRASTA = orth(L_hat_grasta(:, t_1 - t_train + 1 : t_2 - t_train));
                tempGRASTA_SE_Phat_P(mc, jj) = ...
                Calc_SubspaceError(P_track_GRASTA, PP1);   
                end
                
                if(NCRMC == 1)
                P_track_NCRMC = orth(L_hat_ncrmc(:, t_1 - t_train + 1 : t_2 - t_train));
                tempNCRMC_SE_Phat_P(mc, jj) = ...
                Calc_SubspaceError(P_track_NCRMC, PP1);   
                end
                
                if(RPCAGD == 1)
                P_track_RPCAGD = orth(L_hat_RPCAGD(:, t_1 - t_train + 1 : t_2 - t_train));
                tempRPCAGD_SE_Phat_P(mc, jj) = ...
                Calc_SubspaceError(P_track_RPCAGD, PP1);   
                end                
                
            else
                if(NORSTR == 1)
                tempRPCA_SE_Phat_P(mc, jj) = ...
                Calc_SubspaceError(P_track_full_rpca{jj}, PP2);
                end
                
                if(GRASTA == 1)
                P_track_GRASTA = orth(L_hat_grasta(:, t_2 - t_train + 1 : t_max - t_train));
                tempGRASTA_SE_Phat_P(mc, jj) = ...
                Calc_SubspaceError(P_track_GRASTA, PP2);   
                end
                
                if(NCRMC == 1)
                P_track_NCRMC = orth(L_hat_ncrmc(:, t_2 - t_train + 1: t_max - t_train));
                tempNCRMC_SE_Phat_P(mc, jj) = ...
                Calc_SubspaceError(P_track_NCRMC, PP2);   
                end
                
                if(RPCAGD == 1)
                P_track_RPCAGD = orth(L_hat_RPCAGD(:, t_2 - t_train + 1: t_max - t_train));
                tempRPCAGD_SE_Phat_P(mc, jj) = ...
                Calc_SubspaceError(P_track_RPCAGD, PP2);   
                end
                
            end
        end
%         if(tempRPCA_SE_Phat_P(mc,end) > 1e-6)
%             break
%             fprintf('Algo did not converge!')
%         end
fprintf('\n')           
end
if(NORSTR == 1)
errRPCA_L = mean(tempRPCA_err_L, 1);
SE_Phat_P_rpca = mean(tempRPCA_SE_Phat_P, 1);
end
if(NCRMC == 1)
errNCRMC_L(cnt, :) = mean(tempNCRMC_err_L, 1);
SE_Phat_P_NCRMC = mean(tempNCRMC_SE_Phat_P, 1);
end
if(RPCAGD == 1)
errRPCAGD_L(cnt, :) = mean(tempRPCAGD_err_L, 1);
SE_Phat_P_RPCAGD = mean(tempRPCAGD_SE_Phat_P, 1);

end
if(GRASTA == 1)
errGRASTA_L = mean(tempGRASTA_err_L, 1);
SE_Phat_P_GRASTA = mean(tempGRASTA_SE_Phat_P, 1);
end

end_time = clock;
save('RMC_RPCAGD_changingSS_MC100_rhomiss10_rhosparse10.mat');
% 
% str1 = 't';
% str2 = '$$\log SE(\hat{P}, P)$$';
% str3 = ['\rho = ',num2str(rho)];
% str4 = '$$\log \frac{||\hat{l}-l||^2}{||l||^2}$$';
% 
% figure
% 
% % plot(t_calc_pca + t_train,log10(SE_Phat_P_rpca),'*k--','LineWidth',1,'MarkerSize',6)
% hold on
% % plot(t_calc_pca + t_train,log10(SE_Phat_P_GRASTA),'b^--','LineWidth',1,'MarkerSize',6)
% % plot(t_calc_pca + t_train,log10(SE_Phat_P_NCRMC),'ys--','LineWidth',1,'MarkerSize',6)
% plot(t_calc_pca + t_train,log10(SE_Phat_P_RPCAGD),'gs--','LineWidth',1,'MarkerSize',6)
% grid on
% xlabel(str1,'FontSize',14)
% ylabel(str2,'interpreter', 'latex','FontSize',14)
% title(str3);

% figure
% step_size = 1;
% plot(t_train + 1 :step_size: t_max, log10(errRPCA_L(1:step_size:t_max-t_train)),'*b--','LineWidth',1,'MarkerSize',6)
% xlabel(str1,'FontSize',14)
% ylabel(str4,'interpreter', 'latex','FontSize',14)
% title(str3);