function Y = BlockStochPowerMethodGenRank(X, r, B)
%%% This is a prototype function that implements the block stochastic power
%%% method for computing the top singular vector of a matrix in the
%%% streaming data model -- this is general rank case

%%%                         Inputs                          %%%
%%%         x - present column of data matrix               %%%
%%%         B - Block size                                  %%%
%%%         r - actual rank                                 %%%

%%%                         Outputs                         %%%
%%%         Y - estimate of the column space                %%%

[n, d] = size(X);
if(r==1)
    q0 = randn(n, 1);
    q0 = q0 / norm(q0);
    
    for tau = 0 : floor(d / B) - 1
        s = zeros(n, 1);
        for t = B * tau + 1 : B * (tau + 1)
            s = s + 1/B * (q0' * X(:, t)) * X(:, t);
        end
        q0 = s / norm(s);
    end
    Y = q0;
else
    [Q, ~] = qr(randn(n, r));
    Q = Q(:, 1 : r);
    for tau = 0 : floor(d / B) - 1
    %     s = zeros(n, 1);
        S = zeros(n, r);
        for t = B * tau + 1 : B * (tau + 1) 
            %Possible to make it matrix form if non-streaming version
            S = S + 1/B * X(:, t) * (X(:, t)' * Q);
        end
        [Q, ~] = qr(S);
        Q = Q(:, 1 : r);
    end
    Y = Q;
end

end