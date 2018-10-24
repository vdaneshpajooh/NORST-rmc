clear;
clc;
% close all

addpath('YALL1_v1.4')
addpath('PROPACK')
%% Data Generation
n = 1000;
t_max = 8000;
s = ceil(0.05 * n);
t_train = 200;
miss_s = 0;
alpha = 60;
alpha1 = 100;
f = 1;
cnt = 1;
MC = 1;
err_t = zeros(MC, 16);

tic
for mc = 1 : MC
            
            fprintf('Monte-Carlo iteration %d in progress\n', mc);
            
            %%%Generating support set and sparse vectors
%             S = zeros(n, t_max);
%             rho = 1;
%             b0 = 0.1;
%             beta = ceil(b0 * alpha1);
            x_max = 25;
            x_min = 10;
%             alpha1 = 100;
%             num_changes = floor((t_max -t_train)/beta);
%             
%             num_changes1 = floor(alpha1 / beta);
%             
%             flag = 0;
%             ii1 = 1;
%             fval1 = 0;
%             for ii = 1 : num_changes
%                 if(~flag)   %%downward motion
%                     if(ii1 <= num_changes1)
%                         bind = fval1 + (ii1 - 1) * s/rho + 1;
%                         sind = min(bind - 1 + s, n);
%                         ii1 = ii1 + 1;
%                         if(ii1 == num_changes1 + 1)
%                             flag = 1;
%                             ii1 = 1;
%                             fval2 = bind;
%                         end
%                     end
%                 else
%                     if(ii1 <= num_changes1)
%                         bind = max(fval2 - (ii1 - 1) * s/rho , 1);
%                         sind = bind - 1 + s;
%                         ii1 = ii1 + 1;
%                         if(ii1 == num_changes1 + 1)
%                             flag = 0;
%                             ii1 = 1;
%                         end
%                     end
%                 end
%                 idx = bind : sind;
%                 jdx = t_train + (ii-1) * beta + 1 : t_train + ii * beta;
%                 S(idx, jdx) = x_min + ...
%                     (x_max - x_min) * rand(length(idx), beta);
%                 T(idx, jdx) = 1;
%             end
    %%% bernoulli model for outliers
        rho = 0.7; % 1-rho : denotes number of outliers
        BernMat = rand(n, t_max);
        T = 1 .* (BernMat >= rho);
        %imagesc(T)
        length(find(T(:) == 1)) / numel(T)

        S_temp = x_min + (x_max - x_min) * rand(n,t_max);
        S = S_temp .* T;
    
    %%%Generating low-rank matrix
        r_0 = 30;
        r_1 = 0;
        r_2 = 0;
        r = r_0;
        L = zeros(n, t_max);
        
        diag_entries = [linspace(sqrt(f), sqrt(f)/2, r_0)];
        t_1 = t_max;
%         t_2 = 8000;
        coeff_train = zeros(r_0, t_max);
        
        for cc = 1 : r_0
            coeff_train(cc, :) = -diag_entries(cc) + ...
                2 * diag_entries(cc) * rand(1, t_max);
        end
        
        Btemp1 = randn(n);
        B1 = (Btemp1 - Btemp1')/2;
%         Btemp2 = randn(n);
%         B2 = (Btemp2 - Btemp2')/2;
        
        delta1 = .5e-3;
%         delta2 = 0.8 * delta1;
        
        P = orth(randn(n, r_0));
%         PP1 = expm(delta1 * B1)  * P;
%         PP2 = expm(delta2 * B2) * PP1;
        
        L(:, 1:t_1) = P(:, 1:r_0) * coeff_train(:, 1:t_1);
%         L(:, t_1+1:end) = PP1 * coeff_train(:, t_1+1:end);
%         L(:, t_2 + 1 : end) = PP2 * coeff_train(:, t_2+1:end);
        M = L + S;
        
        %% Calls to NORST
        
        %%%Algorithm parameters
        K = 75;
        omega = x_min / 2;
        %     gamma = sqrt(4 * log(n)/n);
        %     s = ceil((gamma + rho) * n);
        
        %%%Call to NORST
        fprintf('NORST\t');
        block_size = ceil(10*log(n));
        ev_thresh = 7.5961e-04;
%         P_init = orth(ncrpca(M(:, 1 : t_train), r, 1e-2, 15));
        P_init = orth(P + 1e-4*randn(n,r));
        [L_hat, P_hat, S_hat, T_hat, t_hat, ...
            P_track_full, t_calc] = ...
            NORST(M(:, t_train + 1 :end),...
            P_init, ev_thresh, alpha, K, omega, block_size);
        
        %%Compute performance metrics
        temp_err_L(mc, :) = ...
            sqrt(mean((L(:, t_train + 1 : end) - L_hat).^2, 1)) ./ ...
            sqrt(mean(L(:, t_train + 1 : end).^2, 1));
        miss_s = ...
            miss_s + (length(find(S_hat))- length(find(S)))/numel(S);

%         Ea1 = orth(PP1(:, end) - (P_init * (P_init'* PP1(:, end))));
            
            %             err_t(mc, cnt) = t_hat(end) + t_train - t_1;
	    %%SS2
            %             Ea2 = orth(PP2(:, end) - (P_init * (P_init'* PP2(:, end))));

        %%Calculate the subspace error
        for jj = 1 : length(t_calc)

                temp_SE_Phat_P(mc, jj) = ...
                Calc_SubspaceError(P_track_full{jj}, P);
        end
            
end

err_L(cnt, :) = mean(temp_err_L, 1);
SE_Phat_P(cnt, :) = mean(temp_SE_Phat_P, 1);
% cnt = cnt + 1;
str2 = '$$\log SE(\hat{P}, P)$$';
plot(t_calc,(log10(SE_Phat_P)))
ylabel(str2,'interpreter', 'latex')
%FigGenReProCS
