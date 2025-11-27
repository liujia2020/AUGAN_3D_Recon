% % ==========================================================
% % Save_VSX_Figures.m (V4 - 最终修复版)
% % 职责:
% % 1. (!! 修复 !!) 强制等待 3 秒，确保 VSX 窗口渲染完成。
% % 2. 查找全部 4 个 VSX 窗口。
% % 3. (!! 修复 !!) 检查坐标轴 (Axes) 是否存在。
% % 4. 将 3 个视图合并保存为 1 个 JPG。
% % 5. (!! 修复 !!) 确保 100% 能关闭所有 4 个 VSX 窗口。
% % ==========================================================
function Save_VSX_Figures()
    
    fprintf('--- 运行 Save_VSX_Figures.m (V4 - 最终修复版) ---\n');
    
    % --- 1. (!! 修复 !!) 强制 MATLAB 刷新并等待 3 秒 ---
    try
        drawnow;
        pause(3.0); % (0.5 -> 3.0) 给予足够的时间渲染
    catch ME_drawnow
        warning('Save_VSX_Figures: drawnow 失败: %s', ME_drawnow.message);
    end

    % --- 2. 定义文件夹和文件名 ---
    MasterOutputDir = 'Ultrasound_Simulation_Data_500_2';
    outputDir = fullfile(MasterOutputDir, '03_VSX_Previews');
    if ~exist(outputDir, 'dir')
        try
            mkdir(outputDir);
            fprintf('已创建新目录: %s\n', outputDir);
        catch ME_mkdir
             warning('Save_VSX_Figures: 创建目录 %s 失败: %s', outputDir, ME_mkdir.message);
        end
    end
    
    % --- 3. 读取通信文件 ---
    comm_file = fullfile(MasterOutputDir, '__latest_config.txt');
    if ~exist(comm_file, 'file')
        warning('Save_VSX_Figures: 未找到通信文件: %s. 无法生成带编号的图像。', comm_file);
        baseName = ['Preview_Unknown_' datestr(now,'HHMMSS')]; % 备用名
    else
        try
            fid = fopen(comm_file, 'r');
            config_filename = fgetl(fid);
            fclose(fid);
            [~, baseName, ~] = fileparts(config_filename);
            baseName = strrep(baseName, 'Config_Scene', 'Preview');
        catch ME_read
            warning('Save_VSX_Figures: 读取通信文件 %s 失败: %s', comm_file, ME_read.message);
            baseName = ['Preview_Error_' datestr(now,'HHMMSS')]; % 备用名
        end
    end
    
    % --- 4. 定义窗口标题 ---
    title_XZ = '3D RC6gV FlashAngles - XZ plane';
    title_YZ = '3D RC6gV FlashAngles - YZ plane';
    title_XY = '3D RC6gV FlashAngles - XY plane';
    title_Control = 'VSX Control';
    
    % --- 5. 查找所有 Figure 句柄 ---
    all_figures = findobj('Type', 'Figure');
    fig_XZ = findobj(all_figures, 'Name', title_XZ);
    fig_YZ = findobj(all_figures, 'Name', title_YZ);
    fig_XY = findobj(all_figures, 'Name', title_XY);
    % (!! 修复 !!) 使用 findall 查找 Java 窗口
    fig_Control = findall(0, 'Name', title_Control); 

    if isempty(fig_XZ) || isempty(fig_YZ) || isempty(fig_XY)
        warning('Save_VSX_Figures: 未能找到所有三个 VSX 视图窗口。取消保存图像。');
    else
        % --- 6. 尝试保存合并的 Figure ---
        try
            % (!! 修复 !!) 检查坐标轴 (Axes) 是否存在
            ax_XZ = findobj(fig_XZ, 'Type', 'Axes');
            ax_YZ = findobj(fig_YZ, 'Type', 'Axes');
            ax_XY = findobj(fig_XY, 'Type', 'Axes');
            
            if isempty(ax_XZ) || isempty(ax_YZ) || isempty(ax_XY)
                error('未能在 Figure 中找到 Axes。VSX 窗口可能未完全渲染。');
            end
            
            % (如果找到多个 Axes，只取第一个)
            ax_XZ = ax_XZ(1);
            ax_YZ = ax_YZ(1);
            ax_XY = ax_XY(1);
            
            hNewFig = figure('Visible', 'off', 'Position', [100 100 1400 450]);

            % --- 子图 1: XZ Plane ---
            new_ax_1 = subplot(1, 3, 1, 'Parent', hNewFig);
            copyobj(allchild(ax_XZ), new_ax_1); 
            set(new_ax_1, 'XLim', get(ax_XZ, 'XLim'), 'YLim', get(ax_XZ, 'YLim'), 'CLim', get(ax_XZ, 'CLim'), 'Colormap', get(ax_XZ, 'Colormap'));
            title(new_ax_1, title_XZ);
            xlabel(new_ax_1, get(get(ax_XZ, 'XLabel'), 'String'));
            ylabel(new_ax_1, get(get(ax_XZ, 'YLabel'), 'String'));
            colorbar(new_ax_1);

            % --- 子图 2: YZ Plane ---
            new_ax_2 = subplot(1, 3, 2, 'Parent', hNewFig);
            copyobj(allchild(ax_YZ), new_ax_2);
            set(new_ax_2, 'XLim', get(ax_YZ, 'XLim'), 'YLim', get(ax_YZ, 'YLim'), 'CLim', get(ax_YZ, 'CLim'), 'Colormap', get(ax_YZ, 'Colormap'));
            title(new_ax_2, title_YZ);
            xlabel(new_ax_2, get(get(ax_YZ, 'XLabel'), 'String'));
            ylabel(new_ax_2, get(get(ax_YZ, 'YLabel'), 'String'));
            colorbar(new_ax_2);

            % --- 子图 3: XY Plane ---
            new_ax_3 = subplot(1, 3, 3, 'Parent', hNewFig);
            copyobj(allchild(ax_XY), new_ax_3); 
            set(new_ax_3, 'XLim', get(ax_XY, 'XLim'), 'YLim', get(ax_XY, 'YLim'), 'CLim', get(ax_XY, 'CLim'), 'Colormap', get(ax_XY, 'Colormap'));
            title(new_ax_3, title_XY);
            xlabel(new_ax_3, get(get(ax_XY, 'XLabel'), 'String'));
            ylabel(new_ax_3, get(get(ax_XY, 'YLabel'), 'String'));
            colorbar(new_ax_3);
            axis(new_ax_3, 'image');
            
            % --- 7. 保存合并后的 Figure ---
            output_filename = fullfile(outputDir, sprintf('%s_Combined.jpg', baseName));
            print(hNewFig, output_filename, '-djpeg', '-r200');
            close(hNewFig); % 关闭这个新创建的 figure
            
            fprintf('已成功保存合并预览图: %s\n', output_filename);
            
        catch ME_copy
            warning('Save_VSX_Figures: 复制/保存图像时出错: %s', ME_copy.message);
            if exist('hNewFig', 'var') && ishandle(hNewFig)
                close(hNewFig);
            end
        end
    end
    
    % --- 8. (!! 修复 !!) 100% 可靠地关闭所有 VSX 窗口 ---
    fprintf('正在关闭 VSX 窗口...\n');
    
    % (!! 修复 !!) 使用 isscalar 和 ishandle 修复 && 崩溃
    if isscalar(fig_XZ) && ishandle(fig_XZ)
        close(fig_XZ);
    end
    if isscalar(fig_YZ) && ishandle(fig_YZ)
        close(fig_YZ);
    end
    if isscalar(fig_XY) && ishandle(fig_XY)
        close(fig_XY);
    end
    if isscalar(fig_Control) && ishandle(fig_Control)
         close(fig_Control);
    end
    
    fprintf('--- Save_VSX_Figures.m (V4) 运行完毕 ---\n');
end