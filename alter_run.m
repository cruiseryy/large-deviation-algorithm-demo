
% parameter

P = [3/4 1/4; 1/2 1/2];
p_inf = (P')^50;
p_inf = p_inf(:,1);

mu = 2;
sigma = 0.5;
T = 200;
k = 1;
dt = 10;

figure(1)
hold on

N = 128;
% pp_a = zeros(100, N);
% amp_a = zeros(100, N);

for rs = 1:20
    tic
    rng(rs+996)
    [pp, amp] = alter_run_func(P, mu, sigma, T, k, dt, N);
%     pp_a(rs, :) = pp;
%     amp_a(rs, :) = amp;
    idx = find(pp < 1);

    lh = scatter(1./pp(idx), amp(idx), 25, 'r', 'filled','MarkerFaceAlpha', 0.1);
%     lh.Color(4)=0.4;
    toc
end

set(gca, 'XScale', 'log')

% 
% tic
% [traj, r] = mc_alter(P, mu, sigma, T, k, dt, N);
% toc
% 
% [n, m] = size(traj);
% Ta = T;
% lambda = 1/Ta*sum(log(r));
% thres = 276.5/100*dt;
% 
% 
% alter_p = zeros(1, m);
% for j = 1:m
%     alter_p(j) = exp(-k/dt*sum(thres - traj(:,j)) + Ta*lambda);
%     pause = 1;
% end
% 
% tot = sum(traj);
% tot2 = sortrows([tot' [1:N]'],1);
% pp = zeros(1, N);
% pp(1) = 1/N*alter_p(tot2(1,2));
% for j = 2:N
%     pp(j) = pp(j-1) + 1/N*alter_p(tot2(j,2));
% end
% 
% figure()
% semilogx(1./pp, 276.5/100*200 - tot2(:,1))