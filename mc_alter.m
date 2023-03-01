function [r, rr] = mc_alter(P, mu, sigma, T, k, dt, N)

p_inf = (P')^50;
p_inf = p_inf(:, 1);

r = zeros(T, 1);
for j = 1:N
    if rand() <= p_inf(1)
        r(1, j) = 0;
    else
        r(1, j) = 1;
    end
end

r = zeros(T, N);
rr = zeros(floor(T/dt), 1);
cur = 0;
for t = 2:T
    for j = 1:N
        if r(t-1, j) == 0
            if rand() <= P(1, 1)
                continue
            else
                r(t, j) = exp(randn()*sigma + mu);
            end
        else
            if rand() <= P(2, 1)
                continue
            else
                r(t, j) = exp(randn()*sigma + mu);
            end
    
        end
    end
    if mod(t, dt) == 0 
        tmp = r(cur+1:cur+dt, :);
        [tmp2, tmpr] = large_value_update(tmp, k);
        rr(1 + floor(cur/dt)) = tmpr;
        r(cur+1:cur+dt, :) = tmp2;
        cur = cur + dt; 
    end
end


end