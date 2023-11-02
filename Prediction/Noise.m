Il%% Prediction of divisive normalization effect given different magnitue of noise
% There are two source of noise, exogenous/input/representation noise and endogenous/output/decision noise
% The representation noise comes with the input, reflecting variability of the
% representation on the value of the item
% The decision noise is intrinsic in the process of decision-making/value
% comparisons. It is noise after the product of value representation.
%% define graphic parameters
mksz = 25;
fontsize = 14;
lwd = 1.5;

%% Distribution of the represented values
% Representational Noise
M = 1;
V1Noise = 50 + 2*randn(1,10000);
V2Noise = 58 + 2*randn(1,10000);
h = figure;
filename = 'Dstrbt_overV3_RprNoise';
subplot(3,2,1); hold on;
pd1 = fitdist(V1Noise','kernel','Kernel','normal');
x = 40:.1:70;
y1 = pdf(pd1,x);
pd2 = fitdist(V2Noise','kernel','Kernel','normal');
y2 = pdf(pd2,x);
plot(x,y1,'k-','LineWidth', lwd);
plot(x,y2,'k-','LineWidth', lwd);
xlabel('Input value');
ylabel('Density');
savefigs(h, filename, './graphics', 14, [4,6]);

V3NoiseL = 30 + 10*randn(1,10000);
V3NoiseL(V3NoiseL<0) = 0;
V3NoiseH = 158 + 10*randn(1,10000);
subplot(3,2,2); hold on;
pd1 = fitdist(V3NoiseL','kernel','Kernel','normal');
x = -20:.1:200;
y1 = pdf(pd1,x);
pd2 = fitdist(V3NoiseH','kernel','Kernel','normal');
y2 = pdf(pd2,x);
plot(x,y1,'b-','LineWidth', lwd);
plot(x,y2,'b-','LineWidth', lwd);
xlabel('V3');
ylabel('Density');
savefigs(h, filename, './graphics', 14, [4,6]);

SV1 = V1Noise./(M + V1Noise + V2Noise + V3NoiseL);
SV2 = V2Noise./(M + V1Noise + V2Noise + V3NoiseL);
subplot(3,1,2); hold on;
pd1 = fitdist(SV1','kernel','Kernel','normal');
x = 0.1:.001:.7;
y1 = pdf(pd1,x);
pd2 = fitdist(SV2','kernel','Kernel','normal');
y2 = pdf(pd2,x);
plot(x,y1,'k-','LineWidth', lwd);
plot(x,y2,'k-','LineWidth', lwd);
area([x(y1>y2), x(y1<y2)],[y2(y1>y2), y1(y2>y1)],'FaceColor','b','FaceAlpha',.5,'EdgeAlpha',.3);
bkp = find(y2 > y1, 1);
pr = sum(y2(x < x(bkp)))/(sum(y2));
text(.245,10, sprintf('%2.1f%% overlap',pr*100), 'FontSize', 18);
xlim([.13, .55]);
xlabel(' ');
ylabel('Density');
savefigs(h, filename, './graphics', 14, [4,6]);

SV1 = V1Noise./(M + V1Noise + V2Noise + V3NoiseH);
SV2 = V2Noise./(M + V1Noise + V2Noise + V3NoiseH);
subplot(3,1,3); hold on;
pd1 = fitdist(SV1','kernel','Kernel','normal');
x = 0.1:.001:.7;
y1 = pdf(pd1,x);
pd2 = fitdist(SV2','kernel','Kernel','normal');
y2 = pdf(pd2,x);
plot(x,y1,'k-','LineWidth', lwd);
plot(x,y2,'k-','LineWidth', lwd);
area([x(y1>y2), x(y1<y2)],[y2(y1>y2), y1(y2>y1)],'FaceColor','b','FaceAlpha',.5,'EdgeAlpha',.3);
bkp = find(y2 > y1, 1);
pr = sum(y2(x < x(bkp)))/(sum(y2));
text(.245,10, sprintf('%2.1f%% overlap',pr*100), 'FontSize', 18);
xlim([.13, .55]);
xlabel('Represented value (R)');
ylabel('Density');
savefigs(h, filename, './graphics', 14, [4,6]);

% Decision Noise
M = 1;
V1Noise = 50 + 0*rand(1,10000);
V2Noise = 58 + 0*rand(1,10000);
h = figure;
filename = 'Dstrbt_overV3_DcsnNoise';
subplot(3,2,1); hold on;
x = 40:.1:70;
y1 = zeros(size(x));
y1(x == 50) = 1;
y2 = zeros(size(x));
y2(x == 58) = 1;
plot(x,y1,'k-','LineWidth', lwd);
plot(x,y2,'k-','LineWidth', lwd);
xlabel('Input value');
ylabel('Density');
savefigs(h, filename, './graphics', 14, [4,6]);

V3NoiseL = 30 + 0*randn(1,10000);
V3NoiseL(V3NoiseL<0) = 0;
V3NoiseH = 158 + 0*randn(1,10000);
subplot(3,2,2); hold on;
x = -20:.1:200;
y1 = zeros(size(x));
y1(x == 30) = 1;
y2 = zeros(size(x));
y2(x == 158) = 1;
plot(x,y1,'r-','LineWidth', lwd);
plot(x,y2,'r-','LineWidth', lwd);
xlabel('V3');
ylabel('Density');
savefigs(h, filename, './graphics', 14, [4,6]);

SV1 = V1Noise./(M + V1Noise + V2Noise + V3NoiseL) + .03*randn(1,10000);
SV2 = V2Noise./(M + V1Noise + V2Noise + V3NoiseL) + .03*randn(1,10000);
subplot(3,1,2); hold on;
pd1 = fitdist(SV1','kernel','Kernel','normal');
x = 0.1:.001:.7;
y1 = pdf(pd1,x);
pd2 = fitdist(SV2','kernel','Kernel','normal');
y2 = pdf(pd2,x);
plot(x,y1,'k-','LineWidth', lwd);
plot(x,y2,'k-','LineWidth', lwd);
area([x(y1>y2), x(y1<y2)],[y2(y1>y2), y1(y2>y1)],'FaceColor','r','FaceAlpha',.5,'EdgeAlpha',.3);
bkp = find(y2 > y1, 1);
pr = sum(y2(x < x(bkp)))/(sum(y2));
text(.25,10, sprintf('%2.1f%% overlap',pr*100), 'FontSize', 18);
xlim([.09, .53]);
xlabel(' ');
ylabel('Density');
savefigs(h, filename, './graphics', 14, [4,6]);

SV1 = V1Noise./(M + V1Noise + V2Noise + V3NoiseH) + .03*randn(1,10000);
SV2 = V2Noise./(M + V1Noise + V2Noise + V3NoiseH) + .03*randn(1,10000);
subplot(3,1,3); hold on;
pd1 = fitdist(SV1','kernel','Kernel','normal');
x = 0.1:.001:.7;
y1 = pdf(pd1,x);
pd2 = fitdist(SV2','kernel','Kernel','normal');
y2 = pdf(pd2,x);
plot(x,y1,'k-','LineWidth', lwd);
plot(x,y2,'k-','LineWidth', lwd);
area([x(y1>y2), x(y1<y2)],[y2(y1>y2), y1(y2>y1)],'FaceColor','r','FaceAlpha',.5,'EdgeAlpha',.3);
bkp = find(y2 > y1, 1);
pr = sum(y2(x < x(bkp)))/(sum(y2));
text(.25,10, sprintf('%2.1f%% overlap',pr*100), 'FontSize', 18);
xlim([.09, .53]);
xlabel('Represented value (R)');
ylabel('Density');
savefigs(h, filename, './graphics', 14, [4,6]);
%% Chosen ratio as a function of V3
M = 50;
K = 75; % sp/s
V1 = 100:10:200;
V2 = 150;
V3 = 0:10:150;
setsz = 400000;
%% Representation noise only
rand('state',2022); % reset random seed
CR = nan([numel(V1), numel(V3)]);
E = nan([numel(V3),1]); % Efficiency
amp_repr = 30;
for v3i = 1:numel(V3)
    for v1i = 1:numel(V1)
       V3Noise = V3(v3i) + ReprNoise(setsz,amp_repr);
       V1Noise = V1(v1i) + ReprNoise(setsz,amp_repr);
       V2Noise = V2 + ReprNoise(setsz,amp_repr);
       
       G1 = V1(v1i) + V2 + V3(v3i) + ReprNoise(setsz,amp_repr) + ReprNoise(setsz,amp_repr) + ReprNoise(setsz,amp_repr);
       G2 = V1(v1i) + V2 + V3(v3i) + ReprNoise(setsz,amp_repr) + ReprNoise(setsz,amp_repr) + ReprNoise(setsz,amp_repr);
       G3 = V1(v1i) + V2 + V3(v3i) + ReprNoise(setsz,amp_repr) + ReprNoise(setsz,amp_repr) + ReprNoise(setsz,amp_repr);
       
       % setting G to be the same value in the denominator arrived the same results.
       % G = V1Noise + V2Noise + V3Noise;
       
       SV1 = K*V1Noise./(M + G1);
       SV2 = K*V2Noise./(M + G2);
       SV3 = K*V3Noise./(M + G3);
       prob = (SV1 > SV2).*(SV1 > SV3) + 2*(SV2 > SV1).*(SV2 > SV3) + 3*(SV3 > SV1).*(SV3 > SV2); % 1: chosen opt 1; 2: chosen opt 2; and 3: chosen opt 3.
       CR(v1i, v3i) =  sum(prob == 1)/ (sum(prob == 1 | prob == 2));
    end
    E(v3i) = mean([CR(V1>V2,v3i); 1 - CR(V1<V2, v3i)], 'omitnan');
end
h = figure; 
filename = 'CR_Representation_Noise';
plot(V3, E, 'b.-','MarkerSize',mksz, 'LineWidth', lwd);
xlabel('V3');
ylabel('% Correct | (1, 2)');
savefigs(h, filename, './graphics', 14, [4,3]);
%% internal noise group
rand('state',2022); % reset random seed
CR = nan([numel(V1), numel(V3)]);
E = nan([numel(V3),1]);
amp_dec = 6;
for v3i = 1:numel(V3)
    V3Noise = V3(v3i);
    for v1i = 1:numel(V1)
       V1Noise = V1(v1i);
       V2Noise = V2;
       
       G1 = V1(v1i) + V2 + V3(v3i);
       G2 = V1(v1i) + V2 + V3(v3i);
       G3 = V1(v1i) + V2 + V3(v3i);
       
       % G = V1(v1i) + V2 + V3(v3i);
       
       SV1 = K*V1Noise./(M + G1) + DecNoise(setsz,amp_dec);
       SV2 = K*V2Noise./(M + G2) + DecNoise(setsz,amp_dec);
       SV3 = K*V3Noise./(M + G3) + DecNoise(setsz,amp_dec);
       prob = (SV1 > SV2).*(SV1 > SV3) + 2*(SV2 > SV1).*(SV2 > SV3) + 3*(SV3 > SV1).*(SV3 > SV2);
       CR(v1i, v3i) =  sum(prob == 1)/ (sum(prob == 1 | prob == 2));
    end
    E(v3i) = mean([CR(V1>V2,v3i); 1 - CR(V1<V2, v3i)], 'omitnan');
end
h = figure;
filename = 'CR_Decision_Noise';
plot(V3, E, 'r.-','MarkerSize',mksz, 'LineWidth', lwd);
xlabel('V3');
ylabel('% Correct | (1, 2)');
savefigs(h, filename, './graphics', 14, [4,3]);

%% mixed noise
M = 1;
% M = 50;
K = 75; % sp/s
V1 = 100:20:200;
V2 = 150;
V3 = 0:20:150;
setsz = 500000;
mycols = [winter(4); flip(autumn(4))];
ampvec = [95 85 75 65 45 35 25 10 % representation
    3 4 5 6 7 8 9 10]; % decision
amp_reprV1V2 = 10;
ldg = [];
h = figure; hold on;
filename = sprintf('CR_Mixed_NoiseV3E%1.1fto%1.1f_I%1.1fto%1.1f',...
    min(ampvec(1,:)), max(ampvec(1,:)), min(ampvec(2,:)), max(ampvec(2,:)));
for ai = 1:length(ampvec)
    amp_repr = ampvec(1,ai);
    amp_dec = ampvec(2,ai);
    rand('state',2022); % reset random seed
    CR = nan([numel(V1), numel(V3)]);
    E = nan([numel(V3),1]);
    for v3i = 1:numel(V3)
        for v1i = 1:numel(V1)
            V3Noise = V3(v3i) + ReprNoise(setsz,amp_repr);
            V1Noise = V1(v1i) + ReprNoise(setsz,amp_reprV1V2);
            V2Noise = V2 + ReprNoise(setsz,amp_reprV1V2);
            
            G1 = V1(v1i) + V2 + V3(v3i) + ReprNoise(setsz,amp_reprV1V2) + ReprNoise(setsz,amp_reprV1V2) + ReprNoise(setsz,amp_repr);
            G2 = V1(v1i) + V2 + V3(v3i) + ReprNoise(setsz,amp_reprV1V2) + ReprNoise(setsz,amp_reprV1V2) + ReprNoise(setsz,amp_repr);
            G3 = V1(v1i) + V2 + V3(v3i) + ReprNoise(setsz,amp_reprV1V2) + ReprNoise(setsz,amp_reprV1V2) + ReprNoise(setsz,amp_repr);
            
            % G = V1Noise + V2Noise + V3Noise;
            
            SV1 = K*V1Noise./(M + G1) + DecNoise(setsz,amp_dec);
            SV2 = K*V2Noise./(M + G2) + DecNoise(setsz,amp_dec);
            SV3 = K*V3Noise./(M + G3) + DecNoise(setsz,amp_dec);
            prob = (SV1 > SV2).*(SV1 > SV3) + 2*(SV2 > SV1).*(SV2 > SV3) + 3*(SV3 > SV1).*(SV3 > SV2);
            CR(v1i, v3i) =  sum(prob == 1)/ (sum(prob == 1 | prob == 2));
        end
        E(v3i) = mean([CR(V1>V2,v3i); 1 - CR(V1<V2, v3i)], 'omitnan');
    end
    ldg(ai) = plot(V3, E, '.-','MarkerSize',mksz/2,'LineWidth',lwd,'Color',mycols(ai,:));
end
legend(ldg, {' ','','','','','','',' '},...
    'FontSize', 7, 'Location', 'eastoutside');
xlabel('V3');
ylabel('% Correct | (1, 2)');
savefigs(h, filename, './graphics', 14, [5.4,4]);
%% mixed noise - 2*2 levels
M = 1;
mycols = [autumn(4); flip(winter(4))];
ampvec = [24    80    24    80 % representation (S L S L)
          8     8    13     13]; % decision (S S L L)
amp_reprV1V2 = 24;
ldg = [];
h = figure; hold on;
filename = sprintf('CR_Mixed_NoiseE%1.1fand%1.1f_I%1.1fand%1.1f',...
    min(amp_repr), max(amp_repr), min(amp_dec), max(amp_dec));
for ai = 1:length(ampvec)
    amp_repr = ampvec(1,ai);
    amp_dec = ampvec(2,ai);
    rand('state',2022); % reset random seed
    CR = nan([numel(V1), numel(V3)]);
    E = nan([numel(V3),1]);
    for v3i = 1:numel(V3)
        for v1i = 1:numel(V1)
            V3Noise = V3(v3i) + ReprNoise(setsz,amp_repr);
            V1Noise = V1(v1i) + ReprNoise(setsz,amp_reprV1V2);
            V2Noise = V2 + ReprNoise(setsz,amp_reprV1V2);
            G1 = V1(v1i) + V2 + V3(v3i) + ReprNoise(setsz,amp_reprV1V2) + ReprNoise(setsz,amp_reprV1V2) + ReprNoise(setsz,amp_repr);
            G2 = V1(v1i) + V2 + V3(v3i) + ReprNoise(setsz,amp_reprV1V2) + ReprNoise(setsz,amp_reprV1V2) + ReprNoise(setsz,amp_repr);
            G3 = V1(v1i) + V2 + V3(v3i) + ReprNoise(setsz,amp_reprV1V2) + ReprNoise(setsz,amp_reprV1V2) + ReprNoise(setsz,amp_repr);
            % G = V1Noise + V2Noise + V3Noise;
            SV1 = K*V1Noise./(M + G1) + DecNoise(setsz,amp_dec);
            SV2 = K*V2Noise./(M + G2) + DecNoise(setsz,amp_dec);
            SV3 = K*V3Noise./(M + G3) + DecNoise(setsz,amp_dec);
            prob = (SV1 > SV2).*(SV1 > SV3) + 2*(SV2 > SV1).*(SV2 > SV3) + 3*(SV3 > SV1).*(SV3 > SV2);
            CR(v1i, v3i) =  sum(prob == 1)/ (sum(prob == 1 | prob == 2));
        end
        E(v3i) = mean([CR(V1>V2,v3i); 1 - CR(V1<V2, v3i)], 'omitnan');
    end
    if ai == 1
        ldg(ai) = plot(V3, E, '.-r','MarkerSize',mksz,'LineWidth',lwd);
    elseif ai == 2
        ldg(ai) = plot(V3, E, '.-b','MarkerSize',mksz,'LineWidth',lwd);
    elseif ai == 3
        ldg(ai) = plot(V3, E, '.-', 'Color', '#ffc0cb','MarkerSize',mksz,'LineWidth',lwd);
    elseif ai == 4
        ldg(ai) = plot(V3, E, '.-', 'Color', '#ADD8E6', 'MarkerSize',mksz,'LineWidth',lwd);
    end
end
legend(ldg, {'','','',''},...
    'FontSize', 7, 'Location', 'eastoutside', 'NumColumns',2);
xlabel('V3');
ylabel('% Correct | (1, 2)');
savefigs(h, filename, './graphics', 14, [5.4,4]);


%% Other possible types of noise for future examinination
V = 1:200;
% Gaussian noise mean-scaled
h = figure; hold on;
plot(V + V.*randn(1,200)/10,V + V.*randn(1,200)/10,'k.');
xlabel('Bid 1');
ylabel('Bid 2');
savefigs(h, 'Input_MeanScaleNoise',   './graphics', 14, [4,4]);
%% Chosen ratio as a function of V3
M = 50;
K = 75; % sp/s
V1 = 100:10:200;
V2 = 150;
V3 = 0:10:150;
setsz = 400000;
%% Representation noise only
rand('state',2022); % reset random seed
CR = nan([numel(V1), numel(V3)]);
E = nan([numel(V3),1]); % Efficiency
amp_repr = 3;
for v3i = 1:numel(V3)
    for v1i = 1:numel(V1)
       V3Noise = V3(v3i) + ReprNoise(setsz,amp_repr)*V3(v3i)/10;
       V1Noise = V1(v1i) + ReprNoise(setsz,amp_repr)*V1(v1i)/10;
       V2Noise = V2 + ReprNoise(setsz,amp_repr)*V2/10;
       
       G1 = V1(v1i) + V2 + V3(v3i) + ReprNoise(setsz,amp_repr)*V1(v1i)/10 + ReprNoise(setsz,amp_repr)*V2/10 + ReprNoise(setsz,amp_repr)*V3(v3i)/10;
       G2 = V1(v1i) + V2 + V3(v3i) + ReprNoise(setsz,amp_repr)*V1(v1i)/10 + ReprNoise(setsz,amp_repr)*V2/10 + ReprNoise(setsz,amp_repr)*V3(v3i)/10;
       G3 = V1(v1i) + V2 + V3(v3i) + ReprNoise(setsz,amp_repr)*V1(v1i)/10 + ReprNoise(setsz,amp_repr)*V2/10 + ReprNoise(setsz,amp_repr)*V3(v3i)/10;
       
       % setting G to be the same value in the denominator arrived the same results.
       % G = V1Noise + V2Noise + V3Noise;
       
       SV1 = K*V1Noise./(M + G1);
       SV2 = K*V2Noise./(M + G2);
       SV3 = K*V3Noise./(M + G3);
       prob = (SV1 > SV2).*(SV1 > SV3) + 2*(SV2 > SV1).*(SV2 > SV3) + 3*(SV3 > SV1).*(SV3 > SV2); % 1: chosen opt 1; 2: chosen opt 2; and 3: chosen opt 3.
       CR(v1i, v3i) =  sum(prob == 1)/ (sum(prob == 1 | prob == 2));
    end
    E(v3i) = mean([CR(V1>V2,v3i); 1 - CR(V1<V2, v3i)], 'omitnan');
end
h = figure; 
filename = 'CR_Representation_MeanScaledNoise';
plot(V3, E, 'b.-','MarkerSize',mksz, 'LineWidth', lwd);
xlabel('V3');
ylabel('% Correct | (1, 2)');
savefigs(h, filename, './graphics', 14, [4,3]);
%% Mixed noise, with representation noise as mean-scaled
M = 1;
% M = 50;
K = 75; % sp/s
V1 = 100:20:200;
V2 = 150;
V3 = 0:20:150;
setsz = 500000;
mycols = [winter(4); flip(autumn(4))];
ampvec = [[95 85 75 65 45 35 25 10]/3; % representation
    3 4 5 6 7 8 9 10]; % decision
amp_reprV1V2 = 10;
ldg = [];
h = figure; hold on;
filename = sprintf('CR_Mixed_MscldNoiseV3E%1.1fto%1.1f_I%1.1fto%1.1f',...
    min(ampvec(1,:)), max(ampvec(1,:)), min(ampvec(2,:)), max(ampvec(2,:)));
for ai = 1:length(ampvec)
    amp_repr = ampvec(1,ai);
    amp_dec = ampvec(2,ai);
    rand('state',2022); % reset random seed
    CR = nan([numel(V1), numel(V3)]);
    E = nan([numel(V3),1]);
    for v3i = 1:numel(V3)
        for v1i = 1:numel(V1)
            V3Noise = V3(v3i) + ReprNoise(setsz,amp_repr)*V3(v3i)/10;
            V1Noise = V1(v1i) + ReprNoise(setsz,amp_reprV1V2)*V1(v1i)/10;
            V2Noise = V2 + ReprNoise(setsz,amp_reprV1V2)*V2/10;
            
            G1 = V1(v1i) + V2 + V3(v3i) + ReprNoise(setsz,amp_reprV1V2)*V1(v1i)/10 + ReprNoise(setsz,amp_reprV1V2)*V2/10 + ReprNoise(setsz,amp_repr)*V3(v3i)/10;
            G2 = V1(v1i) + V2 + V3(v3i) + ReprNoise(setsz,amp_reprV1V2)*V1(v1i)/10 + ReprNoise(setsz,amp_reprV1V2)*V2/10 + ReprNoise(setsz,amp_repr)*V3(v3i)/10;
            G3 = V1(v1i) + V2 + V3(v3i) + ReprNoise(setsz,amp_reprV1V2)*V1(v1i)/10 + ReprNoise(setsz,amp_reprV1V2)*V2/10 + ReprNoise(setsz,amp_repr)*V3(v3i)/10;

            % G = V1Noise + V2Noise + V3Noise;
            
            SV1 = K*V1Noise./(M + G1) + DecNoise(setsz,amp_dec);
            SV2 = K*V2Noise./(M + G2) + DecNoise(setsz,amp_dec);
            SV3 = K*V3Noise./(M + G3) + DecNoise(setsz,amp_dec);
            prob = (SV1 > SV2).*(SV1 > SV3) + 2*(SV2 > SV1).*(SV2 > SV3) + 3*(SV3 > SV1).*(SV3 > SV2);
            CR(v1i, v3i) =  sum(prob == 1)/ (sum(prob == 1 | prob == 2));
        end
        E(v3i) = mean([CR(V1>V2,v3i); 1 - CR(V1<V2, v3i)], 'omitnan');
    end
    ldg(ai) = plot(V3, E, '.-','MarkerSize',mksz/2,'LineWidth',lwd,'Color',mycols(ai,:));
end
legend(ldg, {' ','','','','','','',' '},...
    'FontSize', 7, 'Location', 'eastoutside');
xlabel('V3');
ylabel('% Correct | (1, 2)');
savefigs(h, filename, './graphics', 14, [5.4,4]);
%% Gaussian noise spindle
h = figure; hold on;
plot(V + betapdf(V/200,3,3).*randn(1,200)*5,[V]+betapdf(V/200,3,3).*randn(1,200)*5,'k.');
xlabel('Bid 1');
ylabel('Bid 2');
savefigs(h, 'Input_SpindleNoise',   './graphics', 14, [4,4]);


function e_noise = ReprNoise(setsz,amp)
e_noise = amp*randn(setsz,1);
return;
end

function i_noise = DecNoise(setsz,amp)
i_noise = amp*randn(setsz,1);
return;
end