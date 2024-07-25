%% To examine the double bidding noise in Gluth, 2020
% visualize double bidding
lwd = 1.5;
mksz = 25;
fontsize = 14;

addpath('/Users/bs3667/Documents/MATLAB/Mylib');
datadir = '/Users/bs3667/Dropbox (NYU Langone Health)/Bo Shen Working files/NoiseProject/Gluth2020/ReplicationStudy_Data';
plotdir = '/Users/bs3667/Dropbox (NYU Langone Health)/Bo Shen Working files/NoiseProject/Gluth2020/Analysis/DoubleBid';
filelist = {dir(fullfile(datadir, 'CD*')).name};
for subj = 1:numel(filelist)
    if mod(subj,9) == 1
        h = figure;
        filename = sprintf('DblBidSubID#%ito#%i',subj,subj+8);
    end
    subplot(3,3,9-mod(subj,9)); hold on;
    load(fullfile(datadir, filelist{subj}));
    plot(ContextDepData.BDM.data(:,4),ContextDepData.BDM.data(:,5),'.','MarkerSize',mksz);
    title(['SubID#', filelist{subj}(8:end-4)]);
    xlabel('Bid 1');
    ylabel('Bid 2');
    savefigs(h, filename, './graphics', fontsize, [8,8]);
end

