function [pp, amp] = alter_run_func(P, mu, sigma, T, k, dt, N)
[traj, r] = mc_alter(P, mu, sigma, T, k, dt, N);

[~, m] = size(traj);
Ta = T;
lambda = 1/Ta*sum(log(r));
thres = 276.5/100*dt;

alter_p = zeros(1, m);
for j = 1:m
    alter_p(j) = exp(-k/dt*sum(thres - traj(:,j)) + Ta*lambda);
end

tot = sum(traj);
tot2 = sortrows([tot' [1:N]'],1);
pp = zeros(1, N);
pp(1) = 1/N*alter_p(tot2(1,2));
for j = 2:N
    pp(j) = pp(j-1) + 1/N*alter_p(tot2(j,2));
end

amp = 276.5/100*200 - tot2(:,1);
end