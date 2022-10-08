#pragma once
#include <vector>
#include "aipilot.h"
#include <Windows.h>
#include "onnxruntime_cxx_api.h"


// 具体的智能算法封装类（算法功能在此类中实现）
class AIPilot_TeamIntelligame : public AIPilot
{
public:
	AIPilot_TeamIntelligame(void);
	~AIPilot_TeamIntelligame(void);

public:
	// 初始化
	bool Init(void* pInitData, unsigned short index);

	// 决策一步
	bool Step();

	// 删除
	bool Delete();

	// 更新
	bool IsUpdate();

	// 获取决策输入数据指针
	void* getInput();

	// 获取决策输出数据指针
	void* getOutput();

public:
	// 智能算法输入（即仿真飞机实体向算法模块发送的本机状态及态势数据）
	AIPilotInput	m_Input;

	// 智能算法输出（即算法模块向飞机实体发送的各类控制指令数据）
	AIPilotOutput	m_Output;

private:
	Ort::SessionOptions session_options;
	Ort::AllocatorWithDefaultOptions allocator;
	std::vector<float> m_fStickLat_act{ -1.0,-0.5,0,0.5,1.0};
	std::vector<float> m_fStickLon_act{ -1.0,-0.5,0,0.5,1.0 };
	std::vector<float> m_fThrottle_act{ 0.0,0.5,1.0 };
	std::vector<float> m_fRudder_act{ -1.0,-0.5,0,0.5,1.0 };
	std::vector<Enum_RadarEleScanLine> ScanLine_act{ Enum_RadarEleScanLine_2, Enum_RadarEleScanLine_4};
	std::vector<Enum_RadarAziScanRange> ScanRange_act{ Enum_RadarAziScanRange_30, Enum_RadarAziScanRange_60, Enum_RadarAziScanRange_120 };
	std::vector<bool> m_bWeaponLaunch_act{ true,false };
};
