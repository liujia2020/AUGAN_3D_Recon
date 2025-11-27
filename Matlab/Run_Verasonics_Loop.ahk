; === Verasonics 自动化脚本 (V17 - 全英文日志 + 兼容旧版) ===
;
;   F1 = 启动/运行 自动化循环
;   F2 = 紧急停止并重载脚本
;
;   更新日志 (V17):
;   - (!!) 核心修复: 移除了 'FileEncoding' 命令 (兼容旧版 AHK)
;   - (!!) 核心修复: 将所有日志输出改为英文，彻底避免乱码问题
;   - (!!) 优化: 缩短保存延时为 7 秒
;   - (!!) 修复: 强制 MATLAB 'cd' 到 A_ScriptDir
; ==========================================================

SetTitleMatchMode, 2 ; 窗口标题只需要包含(Contains)以下文字即可

; --- (!! 核心修复 !!) 移除了 FileEncoding (兼容旧版)
; ---

; --- 自动将脚本所在目录设为工作目录 ---
MATLAB_Working_Dir := A_ScriptDir 

; --- 自动构建绝对路径 ---
MasterOutputDir_Relative = Ultrasound_Simulation_Data_500_2 ; (你MATLAB脚本中定义的文件夹)
DebugLogFile = %MATLAB_Working_Dir%\automation_log.txt ; 详细调试日志
ResultsLogFile = %MATLAB_Working_Dir%\successful_pairs_log.csv ; (!! 你要的易读日志 !!)
CommFile = %MATLAB_Working_Dir%\%MasterOutputDir_Relative%\__latest_config.txt

F1::
    FileAppend, --- [%A_Now%] Automation Loop Started (V17) ---`n, %DebugLogFile%
    
    if not FileExist(ResultsLogFile)
    {
        FileAppend, ConfigFile,RF_DataFile`n, %ResultsLogFile%
    }
    
    ; --- 设置你想要生成的总数据量 ---
    TotalRuns := 200 ; (例如: 生成 100 个数据集)
    
    Loop, %TotalRuns%
    {
        LoopStartTime := A_TickCount
        FileAppend, `n[%A_Now%] --- Starting Loop %A_Index% / %TotalRuns% ---`n, %DebugLogFile%

        ; --- 1. 激活MATLAB并强制改变路径 ---
        WinActivate, MATLAB
        Sleep, 1500 
        SendInput, cd '%MATLAB_Working_Dir%'{Enter} ; (!! 强制 CD !!)
        Sleep, 500
        SendInput, clear all{Enter}
        Sleep, 1000

        ; --- 2. 运行 SetUp 脚本 (生成配置) ---
        SendInput, run('SetUpRC6gV_FlashAngles.m'){Enter}
        FileAppend, [%A_Now%] Sent command: SetUpRC6gV_FlashAngles.m...`n, %DebugLogFile%

        ; --- 3. 轮询等待通信文件，最多 60 秒 ---
        WaitLoops := 0
        Loop, 60
        {
            Sleep, 1000 ; (等待 1 秒)
            if FileExist(CommFile)
            {
                FileAppend, [%A_Now%] Comm file detected after %A_Index% seconds.`n, %DebugLogFile%
                break ; (找到文件，跳出循环)
            }
            WaitLoops := A_Index
        }

        if (WaitLoops = 60)
        {
            ; --- SetUp 超时处理 ---
            FileAppend, [%A_Now%] !! ERROR: SetUp script timeout (60s). Comm file not found: %CommFile% !!`n, %DebugLogFile%
            FileAppend, [%A_Now%] SetUp script may be stuck or failed. Cleaning up and skipping loop...`n, %DebugLogFile%
            
            WinActivate, MATLAB
            Sleep, 1000
            SendInput, ^c ; (发送 Ctrl+C)
            Sleep, 500
            
            Continue ; (跳到下一个大循环)
        }

        ; --- 4. 读取通信文件 (现在我们100%确定它存在) ---
        FileRead, ConfigName, %CommFile%
        FileAppend, [%A_Now%] SetUp Success. Read from comm file: %ConfigName%`n, %DebugLogFile%


        ; --- 5. 运行 VSX (启动仿真) ---
        SendInput, VSX{Enter}
        FileAppend, [%A_Now%] Sent command: VSX. Waiting for windows...`n, %DebugLogFile%
        
        ; --- 6. 等待 VSX Control 窗口，超时设为 120 秒 ---
        WinWait, VSX Control, , 120 

        ; --- 7. 检查是否超时 ---
        if ErrorLevel
        {
            ; --- VSX 超时处理 ---
            FileAppend, [%A_Now%] !! ERROR: Loop %A_Index% (%ConfigName%) timed out waiting for 'VSX Control' (120s) !!`n, %DebugLogFile%
            FileAppend, [%A_Now%] Cleaning up and skipping loop...`n, %DebugLogFile%
            
            WinClose, 3D RC6gV FlashAngles - XZ plane
            WinClose, 3D RC6gV FlashAngles - YZ plane
            WinClose, 3D RC6gV FlashAngles - XY plane
            WinClose, VSX Control
            Sleep, 10000

            WinActivate, MATLAB
            Sleep, 1000
            SendInput, ^c ; (发送 Ctrl+C)
            Sleep, 500
            
            Continue 
        }

        ; --- 8. (如果没超时) 等待其他窗口并关闭 ---
        WinWait, 3D RC6gV FlashAngles - XZ plane
        WinWait, 3D RC6gV FlashAngles - YZ plane
        WinWait, 3D RC6gV FlashAngles - XY plane
        FileAppend, [%A_Now%] All VSX windows detected.`n, %DebugLogFile%
        
        Sleep, 15000 ; (保留 8 秒观察时间)

        WinClose, 3D RC6gV FlashAngles - XZ plane
        WinClose, 3D RC6gV FlashAngles - YZ plane
        WinClose, 3D RC6gV FlashAngles - XY plane
        WinClose, VSX Control
        
        ; --- 9. 运行 Save_RF_Data 脚本 (保存结果) ---
        Sleep, 2000 
        WinActivate, MATLAB 
        Sleep, 1000 
        
        SendInput, run('Save_RF_Data.m'){Enter}
        FileAppend, [%A_Now%] Sent command: Save_RF_Data.m`n, %DebugLogFile%
        
        Sleep, 7000 ; (7 秒保存时间)
        
        ; (!! 核心: 写入易读日志 !!)
        StringReplace, RFName, ConfigName, Config_Scene, SimData_RF
        FileAppend, %ConfigName%,%RFName%`n, %ResultsLogFile%

        LoopEndTime := A_LoopTickCount
        LoopDuration := (LoopStartTime - LoopStartTime) / 1000
        FileAppend, [%A_Now%] Loop %A_Index% completed successfully (Duration: %LoopDuration%s)`n, %DebugLogFile%

    }

    ; --- 10. (循环结束) ---
    FileAppend, `n[%A_Now%] --- All %TotalRuns% data acquisition loops completed. ---`n, %DebugLogFile%
    MsgBox, 数据采集循环已完成！
    
return

; --- 按 F2 键紧急停止脚本 ---
F2::
    FileAppend, `n[%A_Znow%] !!! USER EMERGENCY STOP (F2) !!!`n, %DebugLogFile%
    Reload
return