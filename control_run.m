% clear;clc;close all 

% parameter

P = [3/4 1/4; 1/2 1/2];
p_inf = (P')^50;
p_inf = p_inf(:,1);

mu = 2;
sigma = 0.5;
T = 200;
dt = 10;

N = 1e5;
rr = zeros(1, N);
for k = 1:N
    rr(k) = sum(mc_prcp(P, mu , sigma, T));
end

% p_avg = sum(rr);
% p_avg = 276.5;
pp2 = [1:N]/N;
rr = sort(rr);

figure(1)
hold on 
plot(1./pp2, 276.5/100*200 - rr, 'k-', 'LineWidth', 2)

