function [res, rr] = large_value_update(traj, k)


% p_avg = 276.5 for 100 day 
% climatology = 13.83 when dt = 5

[n, m] = size(traj);
res = zeros(n, m);

thres = 276.5/100*n;

weights = zeros(1, m);
for j = 1:m
    weights(j) = exp(k/n*sum(thres - traj(:, j)));
end

rr = mean(weights);
weights = weights/rr;

cdf = zeros(1, m);
cdf(1) = weights(1);
for j = 2:m
    cdf(j) = cdf(j-1) + weights(j);
end
cdf = cdf/cdf(end);
% cdf = cdf + 1/m;
% cdf = cdf/cdf(end);


for j = 1:m
    idx = bisect(cdf, rand());
    res(:, j) = traj(:, idx);
end

end
    