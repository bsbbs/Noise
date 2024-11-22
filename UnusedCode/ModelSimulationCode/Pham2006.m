%% case 1: X has small coefficient of variance, which means most X > 0. But Y not.
rand('state',2023);
X = randn(100000,1)+3;
Y = randn(100000,1);
Z = X./Y;
h = figure;
subplot(3,1,1);
histogram(X);
subplot(3,1,2);
histogram(Y);
subplot(3,1,3);
histogram(Z(Z>= quantile(Z,.1) & Z <= quantile(Z, .9)));

%% case 2: Y has small coefficient of variance, which means most Y > 0. But X not.
rand('state',2023);
X = randn(100000,1);
Y = randn(100000,1)+3;
Z = X./Y;
h = figure;
subplot(3,1,1);
histogram(X);
subplot(3,1,2);
histogram(Y);
subplot(3,1,3);
histogram(Z(Z>= quantile(Z,.1) & Z <= quantile(Z, .9)));
