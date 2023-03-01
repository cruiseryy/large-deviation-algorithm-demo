%% a simple Markov chain to generate a time series of daily precipitation
% P: the transition matrix given by

function r = mc_prcp(P, mu , sigma, T)

p_inf = (P')^50;
p_inf = p_inf(:, 1);

r = zeros(T, 1);
if rand() <= p_inf(1)
    r(1) = 0;
else
    r(1) = 1;
end
for t = 2:T
    if r(t-1) == 0
        if rand() <= P(1, 1)
            continue
        else
            r(t) = exp(randn()*sigma + mu);
        end
    else
        if rand() <= P(2, 1)
            continue
        else
            r(t) = exp(randn()*sigma + mu);
        end

    end
end


end