clear;clc;close all

mu = 2;
sigma = 0.5;

xx = randn(100, 1);
yy = exp(xx*sigma + mu);
figure()
hist(yy)