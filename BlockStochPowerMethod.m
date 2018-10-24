function u = BlockStochPowerMethod(X, B)
%%% This is a prototype function that implements the block stochastic power
%%% method for computing the top singular vector of a matrix in the
%%% streaming data model

%%%                         Inputs                          %%%
%%%         x - present column of data matrix               %%%
%%%         B - Block size                                  %%%
%%%         tau - index of some sort                        %%%

%%%                         Outputs                         %%%
%%%         y - current estimate of singular vector         %%%
%%%         t - time index                                  %%%


[n, d] = size(X);
q0 = randn(n, 1);
q0 = q0 / norm(q0);


for tau = 0 : floor(d / B) - 1
    s = zeros(n, 1);
    for t = B * tau + 1 : B * (tau + 1)
        s = s + 1/B * (q0' * X(:, t)) * X(:, t);
    end
    q0 = s / norm(s);
end
u = q0;
end
