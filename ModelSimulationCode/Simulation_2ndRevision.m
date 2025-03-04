%% 
%% define directories
DefineIO;
%% Loading the data transformed in the code: /Users/bs3667/Noise/modelfit/ModelFit-DataTrnsfrm.m
load(fullfile(Gitdir, 'myData', 'TrnsfrmData.mat'), 'mt');
Simdir = fullfile(rootdir, 'Prediction/Revision2');
plot_dir = fullfile(Simdir, 'plot');
if ~exist(plot_dir,'dir')
    mkdir(Simdir);
    mkdir(plot_dir);
end
%% Transform data
blacklist = [22102405; 22102705; 22102708; 22071913; 22110306];
sublist = unique(mt.subID);
sublist = sublist(~ismember(sublist, blacklist));
fulllist = unique(mt.subID);
N = length(sublist);
mtconvert = [];
for s = 1:N
    indvtask = mt(mt.subID == sublist(s),:);
    Vtrgt = unique([indvtask.V1; indvtask.V2]);
    mintrgt = min(Vtrgt);
    if mintrgt > 0 % skip this subject if the min target value is zero, because the values cannot be scaled and the value space does not help in testing of the hypothesis
        indvtask.V1scld = indvtask.V1/mintrgt;
        indvtask.V2scld = indvtask.V2/mintrgt;
        indvtask.V3scld = indvtask.V3/mintrgt;
        indvtask.sdV1scld = indvtask.sdV1/mintrgt;
        indvtask.sdV2scld = indvtask.sdV2/mintrgt;
        indvtask.sdV3scld = indvtask.sdV3/mintrgt;
        mtconvert = [mtconvert; indvtask];
    end  
end
mtconvert.choice = mtconvert.chosenItem - 1;
mtconvert = mtconvert(~isnan(mtconvert.chosenItem) & mtconvert.V1 ~= mtconvert.V2,:);
V1mean = mean(mtconvert.V1scld)*83; %128.4661; %88; %88;
V2mean = mean(mtconvert.V2scld)*83; % 244.5383
V3 = mtconvert.V3scld*V2mean;
sdV3 = mtconvert.sdV3scld*V2mean;
eps1 = mean(mtconvert.sdV1scld)*83; %48.4466; % early noise for V1
eps2 = mean(mtconvert.sdV2scld)*83; %76.7159; % early noise for V2
V1 = V1mean*ones(size(V3));
V2 = V2mean*ones(size(V3));
sdV1 = eps1*ones(size(V3));
sdV2 = eps2*ones(size(V3));
mtconvert.V1 = V1;
mtconvert.V2 = V2;
mtconvert.V3 = V3;
mtconvert.sdV1 = sdV1;
mtconvert.sdV2 = sdV2;
mtconvert.sdV3 = sdV3;
etavec = linspace(0,15,6);
etatxt = cellstr(string(etavec));
%% Simulation
for modeli = 4
    fprintf('Model %d:\t', modeli);
    modelname = sprintf('Model%i_Predict_V1%i_V2%i_sd1%1.1f_etavec%i', modeli, V1mean, V2mean, eps1, numel(etavec));
    simdat = fullfile(Simdir, [modelname, '.mat']);
    if ~exist(simdat, 'file')
        mtmodel = [];
        for etai = 1:numel(etavec)
            eta = etavec(etai);
            fprintf('\nEta %d:\t', eta);
            dat = mtconvert;
            Mp = 1;
            switch modeli
                case 1
                    %%
                    x = [Mp, eta];
                    probs = McFadden(x, dat);
                    name = 'McFadden';
                case 2
                    scl = 1; %fit.scl(subjmask & fit.modeli == modeli);
                    x = [Mp, eta, scl];
                    probs = Mdl2(x, dat);
                    name = 'LinearDistrb';
                case 3
                    wp = 1; %fit.wp(subjmask & fit.modeli == modeli);
                    x = [Mp, eta, wp];
                    probs = DN(x, dat);
                    name = 'DN'; %, cut input, independent';
                case 4
                    scl = 1; %fit.scl(subjmask & fit.modeli == modeli);
                    wp = 1; %fit.wp(subjmask & fit.modeli == modeli);
                    x = [Mp, eta, wp, scl];
                    probs = dDNb(x, dat, 'absorb');
                    name = 'dDNb'; %, cut input, independent';
            end
            dat.modelprob1 = gather(probs(:,1));
            dat.modelprob2 = gather(probs(:,2));
            dat.modelprob3 = gather(probs(:,3));
            dat.ratio = dat.modelprob2./(dat.modelprob1 + dat.modelprob2);
            dat.eta = eta*ones(size(dat.V1));
            mtmodel =  [mtmodel; dat];
            fprintf('\n');
        end
        save(simdat, "mtmodel", '-mat');
        writetable(mtmodel, fullfile(Simdir, [modelname, '.txt']), 'Delimiter', '\t');
    else
        load(simdat);
    end

    %% Visualize in sliding windows
    dat = mtmodel(mtmodel.chosenItem ~= 3,:);
    GrpMean = grpstats(dat, ["subID", "eta", "ID3"], "mean", "DataVars", ["V3", "sdV3", "V3scld", "sdV3scld", "choice","ratio"]);
    colorpalette = flip(jet(6));
    Window = 0.15;
    LowestV3 = 0; %0.2;
    HighestV3 = 1; %.8;
    h = figure;
    filename = sprintf('%s_Predict', modelname);
    hold on;
    lgd = [];
    for etai = 1:numel(etavec)
        eta = etavec(etai);
        Ntrial = [];
        choice = [];
        choicese = [];
        ratio = [];
        ratiose = [];
        sdV3scld = [];
        v3vec = LowestV3:.015:HighestV3;
        dat = GrpMean(GrpMean.eta == eta & GrpMean.mean_V3scld >= LowestV3 &  GrpMean.mean_V3scld <= HighestV3,:);
        for v3 = v3vec
            section = dat(dat.mean_V3scld >= v3 - Window & dat.mean_V3scld <= v3 + Window,:);
            Ntrial = [Ntrial, sum(section.GroupCount)];
            % choice = [choice, sum(section.mean_choice.*section.GroupCount)/sum(section.GroupCount)];
            % choicese = [choicese, std(section.mean_choice)/sqrt(length(section.mean_choice))];
            ratio = [ratio, sum(section.mean_ratio.*section.GroupCount)/sum(section.GroupCount)];
            % ratiose = [ratiose, std(section.mean_ratio)/sqrt(length(section.mean_ratio))];
            sdV3scld = [sdV3scld, sum(section.mean_sdV3scld.*section.GroupCount)/sum(section.GroupCount)];
        end
        cut = Ntrial > 100;
        % scatter(v3vec(cut), ratio(cut), Ntrial(cut)/80*5, 'color', colorpalette{i});
        % plot(v3vec(cut), choice(cut)*100, '-', 'Color', colorpalette{i}, 'LineWidth', 2);
        lgd(etai) = plot(v3vec(cut), ratio(cut)*100, '-', 'Color', colorpalette(etai,:), 'LineWidth', 2);
        % fill([v3vec fliplr(v3vec)], [ratio-ratiose fliplr(ratio+ratiose)], rgbMatrix(vi,:), 'FaceAlpha', 0.3, 'EdgeColor', 'none');
    end
    xlim([LowestV3, HighestV3]);
    xlabel('Scaled V3');
    ylabel('% Correct | V1, V2');
    l = legend(lgd, {'0','3', '6','9','12','15'}, 'Location','northeast');
    title(l, 'Late noise');
    mysavefig(h, filename, plot_dir, 12, [4, 4]);
end


% %% Simulation
% for modeli = 4
%     modelname = sprintf('Model%i_Predict_V1%i_V2%i_sd1%1.1f', modeli, V1mean, V2mean, eps1);
%     simdat = fullfile(Simdir, [modelname, '.mat']);
%     if ~exist(simdat, 'file')
%         fprintf('Model %d:\t', modeli);
%         dat = mtconvert;
%         %subjmask = fit.subID == find(fulllist == sublist(s));
%         %Mp = fit.Mp(subjmask & fit.modeli == modeli);
%         %delta = fit.delta(subjmask & fit.modeli == modeli);
%         Mp = 1;
%         delta = 0.428;
%         switch modeli
%             case 1
%                 %%
%                 x = [Mp, delta];
%                 probs = McFadden(x, dat);
%                 name = 'McFadden';
%             case 2
%                 scl = 1; %fit.scl(subjmask & fit.modeli == modeli);
%                 x = [Mp, delta, scl];
%                 probs = Mdl2(x, dat);
%                 name = 'LinearDistrb';
%             case 3
%                 wp = 1; %fit.wp(subjmask & fit.modeli == modeli);
%                 x = [Mp, delta, wp];
%                 probs = DN(x, dat);
%                 name = 'DN'; %, cut input, independent';
%             case 4
%                 scl = 1; %fit.scl(subjmask & fit.modeli == modeli);
%                 wp = 1; %fit.wp(subjmask & fit.modeli == modeli);
%                 x = [Mp, delta, wp, scl];
%                 probs = dDNb(x, dat, 'absorb');
%                 name = 'dDNb'; %, cut input, independent';
%         end
%         dat.modelprob1 = gather(probs(:,1));
%         dat.modelprob2 = gather(probs(:,2));
%         dat.modelprob3 = gather(probs(:,3));
%         mtmodel =  dat;
%         fprintf('\n');
%         mtmodel.ratio = mtmodel.modelprob1./(mtmodel.modelprob1 + mtmodel.modelprob2);
%         save(simdat, "mtmodel", '-mat');
%         writetable(mtmodel, fullfile(Simdir, [modelname, '.txt']), 'Delimiter', '\t');
%     else
%         load(simdat);
%     end
%     %% Visualize in sliding windows
%     dat = mtmodel(mtmodel.chosenItem ~= 3,:);
%     GrpMean = grpstats(dat, ["subID", "TimeConstraint", "Vaguenesscode", "ID3"], "mean", "DataVars", ["V3", "sdV3", "V3scld", "sdV3scld", "choice","ratio"]);
%     colorpalette ={'r','#FFBF00','#00FF80','b'};
%     rgbMatrix = [
%         0, 0, 255;   % Blue
%         255, 192, 203; % Pink
%         173, 216, 230; % Light Blue
%         255, 0, 0     % Red
%         ]/255;
%     Window = 0.15;
%     LowestV3 = 0; %0.2;
%     HighestV3 = 1; %.8;
%     h = figure;
%     filename = sprintf('%s_Predict[Full]', modelname);
%     vi = 0;
%     vtext = {'Vague','Precise'};
%     i = 0;
%     for v = [1, 0] % vague, precise
%         vi = vi + 1;
%         subplot(1,2,vi); hold on;
%         ti = 0;
%         for t = [10, 1.5] % low, high
%             i = i + 1;
%             ti = ti + 1;
%             Ntrial = [];
%             choice = [];
%             choicese = [];
%             ratio = [];
%             ratiose = [];
%             sdV3scld = [];
%             v3vec = LowestV3:.015:HighestV3;
%             dat = GrpMean(GrpMean.TimeConstraint == t & GrpMean.Vaguenesscode == v & GrpMean.mean_V3scld >= LowestV3 &  GrpMean.mean_V3scld <= HighestV3,:);
%             for v3 = v3vec
%                 section = dat(dat.mean_V3scld >= v3 - Window & dat.mean_V3scld <= v3 + Window,:);
%                 Ntrial = [Ntrial, sum(section.GroupCount)];
%                 choice = [choice, sum(section.mean_choice.*section.GroupCount)/sum(section.GroupCount)];
%                 % choicese = [choicese, std(section.mean_choice)/sqrt(length(section.mean_choice))];
%                 ratio = [ratio, sum(section.mean_ratio.*section.GroupCount)/sum(section.GroupCount)];
%                 % ratiose = [ratiose, std(section.mean_ratio)/sqrt(length(section.mean_ratio))];
%                 sdV3scld = [sdV3scld, sum(section.mean_sdV3scld.*section.GroupCount)/sum(section.GroupCount)];
%             end
%             cut = Ntrial > 100;
%             % scatter(v3vec(cut), ratio(cut), Ntrial(cut)/80*5, 'color', colorpalette{i});
%             % plot(v3vec(cut), choice(cut)*100, '-', 'Color', colorpalette{i}, 'LineWidth', 2);
%             lgd(ti) = plot(v3vec(cut), ratio(cut)*100, '-', 'Color', colorpalette{i}, 'LineWidth', 2);
%             % fill([v3vec fliplr(v3vec)], [ratio-ratiose fliplr(ratio+ratiose)], rgbMatrix(vi,:), 'FaceAlpha', 0.3, 'EdgeColor', 'none');
%         end
%         xlim([LowestV3, HighestV3]);
%         xlabel('Scaled V3');
%         ylabel('% Correct | V1, V2');
%         title(vtext{vi});
%         l = legend(lgd, {'Low','High'}, 'Location','northeast');
%         title(l, 'Time pressure');
%         mysavefig(h, filename, plot_dir, 12, [8, 4]);
%     end
%     %% Visualization in heatmap
%     dat = mtmodel(mtmodel.chosenItem ~= 3,:);
%     GrpMean = grpstats(dat, ["TimeConstraint", "Vaguenesscode", "ID3"], "mean", "DataVars", ["V3", "sdV3", "V3scld", "sdV3scld", "choice", "ratio"]);
%     Window = 0.15;
%     Varrng = [min(GrpMean.mean_sdV3scld), .4];% max(GrpMean.mean_sdV3scld)];
%     Bindow = 0.15/2;
%     h = figure;
%     filename = sprintf('%s_Heatmap', modelname);
%     ti = 0;
%     TimePressure = {'Low','High'};
%     for t = [10, 1.5] % low, high
%         ti = ti + 1;
%         dat = GrpMean(GrpMean.TimeConstraint == t,:);
%         v3vec = LowestV3:.03:HighestV3;
%         varvec = Varrng(1):.015:Varrng(2);
%         Ntrial = NaN(numel(varvec), numel(v3vec));
%         ratio = NaN(numel(varvec), numel(v3vec));
%         ratiose = NaN(numel(varvec), numel(v3vec));
%         sdV3scld = NaN(numel(varvec), numel(v3vec));
%         for vi = 1:numel(v3vec)
%             for ri = 1:numel(varvec)
%                 v3 = v3vec(vi);
%                 r = varvec(ri);
%                 maskv3 = dat.mean_V3scld >= v3 - Window & dat.mean_V3scld <= v3 + Window;
%                 maskr3 = dat.mean_sdV3scld >= r - Bindow & dat.mean_sdV3scld <= r + Bindow;
%                 section = dat(maskv3 & maskr3,:);
%                 Ntrial(ri,vi) = sum(section.GroupCount);
%                 ratio(ri,vi) = mean(section.mean_ratio);
%                 ratiose(ri,vi) = std(section.mean_ratio)/sqrt(length(section.mean_ratio));
%                 sdV3scld(ri,vi) = mean(section.mean_sdV3scld);
%             end
%         end
%         ratio(Ntrial<50) = NaN;
%         subplot(2, 2, 1+(ti-1)*2); hold on;
%         colormap("bone");
%         cmap = bone(numel(varvec));
%         for ri = 1:numel(varvec)
%             plot(v3vec, ratio(ri,:), '.-', 'Color', cmap(ri,:));
%         end
%         title(TimePressure{ti});
%         xlabel('Scaled V3');
%         ylabel('% Correct | V1 & V2');
%         mysavefig(h, filename, plot_dir, 12, [9, 8]);
% 
%         subplot(2, 2, 2+(ti-1)*2); hold on;
%         colormap("jet");
%         imagesc(v3vec, varvec, ratio);
%         c = colorbar('Location', 'northoutside');
%         % ylim([0,1]);
%         ylabel(c, '% Correct | V1 & V2');
%         xlabel('Scaled V3');
%         ylabel('V3 Variance');
%         mysavefig(h, filename, plot_dir, 12, [9, 8]);
%     end
% end

%% 
function probs = McFadden(x, dat)
if gpuDeviceCount > 0
    gpuparallel = 1;
else
    gpuparallel = 0;
end
Mp = x(1); % change M to be M', absorbing the magnitude of late noise
%eta = 1; % after the transformation, the late noise term is standardized as 1
eta = x(2); % late noise difference between time-pressure conditions
delta = 0;
data = dat(:, {'V1', 'V2', 'V3', 'sdV1','sdV2','sdV3','chosenItem','TimeConstraint'});
num_samples = 20000;
samples = [];
for ci = 1:3
    if gpuparallel
        values = gpuArray(data.(['V',num2str(ci)])');
    else
        values = data.(['V',num2str(ci)])';
    end
    samples(:,:,ci) = repmat(values, num_samples, 1);
end
if gpuparallel
    SVs = samples/Mp + gpuArray.randn(size(samples)).*(1 + delta*repmat(data.TimeConstraint'==1.5,num_samples,1,3))*eta;
    choice = gpuArray(data.chosenItem');
else
    SVs = samples/Mp + randn(size(samples)).*(1 + delta*repmat(data.TimeConstraint'==1.5,num_samples,1,3))*eta;
    choice = data.chosenItem';
end
max_from_each_distribution = SVs == max(SVs, [], 3);
probs = squeeze(sum(max_from_each_distribution, 1) / size(SVs, 1));
% nll = -sum(log(max(probs(sub2ind(size(probs), 1:size(probs, 1), choice)), eps)));
% if gpuparallel
%     nll = gather(nll);
% end
end

function probs = Mdl2(x, dat)
if gpuDeviceCount > 0
    gpuparallel = 1;
else
    gpuparallel = 0;
end
Mp = x(1); % change M to be M', absorbing the magnitude of late noise
% eta = 1; % after the transformation, the late noise term is standardized as 1
eta = x(2); % late noise difference between time-pressure conditions
delta = 0;
scl = x(3); % scaling parameter on early noise
data = dat(:, {'V1', 'V2', 'V3', 'sdV1','sdV2','sdV3','chosenItem','TimeConstraint'});
num_samples = 20000;
Ntrl = size(dat,1);
samples = [];
for ci = 1:3
    if gpuparallel
        values = gpuArray(data.(['V',num2str(ci)])');
        stds = gpuArray(data.(['sdV', num2str(ci)])')*scl;
        samples(:,:,ci) = gpuArray.randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1);
    else
        values = data.(['V',num2str(ci)])';
        stds = data.(['sdV', num2str(ci)])'*scl;
        samples(:,:,ci) = randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1);
    end
end
if gpuparallel
    SVs = samples/Mp + gpuArray.randn(size(samples)).*(1 + delta*repmat(data.TimeConstraint'==1.5,num_samples,1,3))*eta;
    choice = gpuArray(data.chosenItem');
else
    SVs = samples/Mp + randn(size(samples)).*(1 + delta*repmat(data.TimeConstraint'==1.5,num_samples,1,3))*eta;
    choice = data.chosenItem';
end
max_from_each_distribution = SVs == max(SVs, [], 3);
probs = squeeze(sum(max_from_each_distribution, 1) / size(SVs, 1));
% nll = -sum(log(max(probs(sub2ind(size(probs), 1:size(probs, 1), choice)), eps)));
% if gpuparallel
%     nll = gather(nll);
% end
end

function probs = DN(x, dat)
if gpuDeviceCount > 0
    gpuparallel = 1;
else
    gpuparallel = 0;
end
Mp = x(1); % change M to be M', absorbing the magnitude of late noise
% eta = 1; % after the transformation, the late noise term is standardized as 1
eta = x(2); % late noise difference between time-pressure conditions
delta = 0;
wp = x(3); % do the same transformation on w
data = dat(:, {'V1', 'V2', 'V3', 'sdV1','sdV2','sdV3','chosenItem','TimeConstraint'});
num_samples = 20000;
samples = [];
for ci = 1:3
    if gpuparallel
        values = gpuArray(data.(['V',num2str(ci)])');
    else
        values = data.(['V',num2str(ci)])';
    end
    samples(:,:,ci) = repmat(values, num_samples, 1);
end
% The product of divisive normalization before adding late noise
DNP = samples./(sum(samples, 3)*wp + Mp);
% adding late noise
if gpuparallel
    SVs = DNP + gpuArray.randn(size(samples)).*(1 + delta*repmat(data.TimeConstraint'==1.5,num_samples,1,3))*eta;
    choice = gpuArray(data.chosenItem');
else
    SVs = DNP + randn(size(samples)).*(1 + delta*repmat(data.TimeConstraint'==1.5,num_samples,1,3))*eta;
    choice = data.chosenItem';
end
max_from_each_distribution = SVs == max(SVs, [], 3);
probs = squeeze(sum(max_from_each_distribution, 1) / size(SVs, 1));
% nll = -sum(log(max(probs(sub2ind(size(probs), 1:size(probs, 1), choice)), eps)));
% if gpuparallel
%     nll = gather(nll);
% end
end

function probs = dDNb(x, dat, mode) % cut inputs, independent
% set the lower boundary for every input value distribution as zero
% samples in the denominator are independent from the numerator
% the SIGMA term in the denominator is non-negative after that.
if gpuDeviceCount > 0
    gpuparallel = 1;
else
    gpuparallel = 0;
end
Mp = x(1); % change M to be M', absorbing the magnitude of late noise
% eta = 4; % after the transformation, the late noise term is standardized as 1
eta = x(2); % late noise difference between time-pressure conditions
delta = 0;
wp = x(3); % do the same transformation on w
scl = x(4); % scaling parameter on the early noise
data = dat(:, {'V1', 'V2', 'V3', 'sdV1','sdV2','sdV3','chosenItem','TimeConstraint'});
num_samples = 20000;
Ntrl = size(dat,1);
if strcmp(mode, 'absorb')
    samples = [];
    for ci = 1:3
        if gpuparallel
            values = gpuArray(data.(['V',num2str(ci)])');
            stds = gpuArray(data.(['sdV', num2str(ci)])')*scl;
            samples(:,:,ci) = max(gpuArray.randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1), 0);
        else
            values = data.(['V',num2str(ci)])';
            stds = data.(['sdV', num2str(ci)])'*scl;
            samples(:,:,ci) = max(randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1), 0);
        end
    end
    D1 = [];
    D2 = [];
    D3 = [];
    for ci = 1:3
        if gpuparallel
            values = gpuArray(data.(['V',num2str(ci)])');
            stds = gpuArray(data.(['sdV', num2str(ci)])')*scl;
            D1(:,:,ci) = max(gpuArray.randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1), 0);
            D2(:,:,ci) = max(gpuArray.randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1), 0);
            D3(:,:,ci) = max(gpuArray.randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1), 0);
        else
            values = data.(['V',num2str(ci)])';
            stds = data.(['sdV', num2str(ci)])'*scl;
            D1(:,:,ci) = max(randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1), 0);
            D2(:,:,ci) = max(randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1), 0);
            D3(:,:,ci) = max(randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1), 0);
        end
    end
elseif strcmp(mode, 'cutoff')
    error('The cutoff boundary algorithm has not been developped yet.');
end
D1 = sum(D1, 3)*wp + Mp;
D2 = sum(D2, 3)*wp + Mp;
D3 = sum(D3, 3)*wp + Mp;
K = 75; % the neural maximum response as assumed
% The product of divisive normalization before adding late noise
DNP = K.*samples./cat(3, D1, D2, D3);
clear D1 D2 D3;
if gpuparallel
    SVs = DNP + gpuArray.randn(size(samples)).*(1 + delta*repmat(data.TimeConstraint'==1.5,num_samples,1,3))*eta;
    choice = gpuArray(data.chosenItem');
else
    SVs = DNP + randn(size(samples)).*(1 + delta*repmat(data.TimeConstraint'==1.5,num_samples,1,3))*eta;
    choice = data.chosenItem';
end
max_from_each_distribution = SVs == max(SVs, [], 3);
probs = squeeze(sum(max_from_each_distribution, 1) / size(SVs, 1));
end
