clear; close all;

df = readtable('box_behken.csv');
mat = table2array(df);
X = mat(:,2:end-1); %remove the first and last column
y = mat(:,end); %y is the last column

beta = inv(transpose(X)*X)*transpose(X)*y;

% beta = [-1.14 1.26 1.23 -0.72 0.97...
%        -1.05 -0.85 0.28 0.20 -4.20 1.93 ...
%        3.65 0.43 0.50 -1.10 -2.13 -5.73 -6.31 -2.91...
%        -3.53 59.15]';

f = @(x1,x2,x3,x4,x5) ...
    beta(1)*x1 + beta(2)*x2 + beta(3)*x3 + ...
    beta(4)*x4 + beta(5)*x5 + beta(6)*x1.*x2 + ...
    beta(7)*x1.*x3 + beta(8)*x1.*x4 + beta(9)*x1.*x5 + ...
    beta(10)*x2.*x3 + beta(11)*x2.*x4 + beta(12)*x2.*x5 + ...
    beta(13)*x3.*x4 + beta(14)*x3.*x5 + beta(15)*x4.*x5;


mask_table = readtable('mask.csv');
mask_array = table2array(mask_table);
mask_array = mask_array(:,2:end);
tmp = size(mask_array);
N = tmp(1);

for idx=1:N
    subplot(2,N/2,idx);
    % x1 = linspace(0,1);
    % x2 = linspace(5,15);
    % x3 = linspace(0.1,0.3);
    % x4 = linspace(2,4);
    % x5 = linspace(0.1,0.3);
    x1 = linspace(-1,1);
    x2 = x1;
    x3 = x1;
    x4 = x1;
    x5 = x1;
    switch(idx)
        case 1
            [X1,X2] = meshgrid(x1,x2);
            Z = f(X1,X2,0,0,0);
            contourf(X1,X2,Z);
        case 2
            [X1,X3] = meshgrid(x1,x3);
            Z = f(X1,0,X3,0,0);
            contourf(X1,X3,Z);
        case 3
            [X1,X4] = meshgrid(x1,x4);
            Z = f(X1,0,0,X4,0);
            contourf(X1,X4,Z);
        case 4
            [X1,X5] = meshgrid(x1,x5);
            Z = f(X1,0,0,0,X5);
            contourf(X1,X5,Z);
        case 5
            [X2,X3] = meshgrid(x2,x3);
            Z = f(0,X2,X3,0,0);
            contourf(X2,X3,Z);
        case 6
            [X2,X4] = meshgrid(x2,x4);
            Z = f(0,X2,0,X4,0);
            contourf(X2,X4,Z);
        case 7
            [X2,X5] = meshgrid(x2,x5);
            Z = f(0,X2,0,0,X5);
            contourf(X2,X5,Z);
        case 8
            [X3,X4] = meshgrid(x3,x4);
            Z = f(0,0,X3,X4,0);
            contourf(X3,X4,Z);
        case 9
            [X3,X5] = meshgrid(x3,x5);
            Z = f(0,0,X3,0,X5);
            contourf(X3,X5,Z);
        case 10
            [X4,X5] = meshgrid(x4,x5);
            Z = f(0,0,0,X4,X5);
            contourf(X4,X5,Z);
    end
end

big;



