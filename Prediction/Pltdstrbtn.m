function h = Pltdstrbtn(Vtrgt, V3, plot_dir, filename)
colorpalette = {'#ef476f','#ffd166','#06d6a0','#118ab2','#073b4c'};
lwd = 2;
fontsize = 11;
Mbins = 44;
Bwd = .1;
h = figure;
hold on;
mydata = Vtrgt(~isnan(Vtrgt) & Vtrgt < Inf);
hst1 = histogram(mydata,...
     'EdgeColor', 'none', 'FaceColor', colorpalette{4}, 'FaceAlpha', .3, 'Normalization', 'pdf');
pd1 = fitdist(mydata,'kernel','Kernel','normal');
x = hst1.BinEdges;
y1 = pdf(pd1,x);
plot(x,y1, '-', 'Color', colorpalette{4}, 'LineWidth', lwd);

mydata = V3(~isnan(V3) & V3 < Inf);
hst2 = histogram(mydata,...
     'EdgeColor', 'none', 'FaceColor', colorpalette{1}, 'FaceAlpha', .3, 'Normalization', 'pdf');
pd2 = fitdist(mydata,'kernel','Kernel','normal');
x = hst2.BinEdges;
y2 = pdf(pd2,x);
plot(x,y2, '-', 'Color', colorpalette{1}, 'LineWidth', lwd);
mysavefig(h, filename, plot_dir, fontsize, [6, 6]);
end