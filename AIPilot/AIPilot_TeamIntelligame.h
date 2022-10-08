#pragma once
#include <vector>
#include "aipilot.h"
#include <Windows.h>
#include "onnxruntime_cxx_api.h"


// ����������㷨��װ�ࣨ�㷨�����ڴ�����ʵ�֣�
class AIPilot_TeamIntelligame : public AIPilot
{
public:
	AIPilot_TeamIntelligame(void);
	~AIPilot_TeamIntelligame(void);

public:
	// ��ʼ��
	bool Init(void* pInitData, unsigned short index);

	// ����һ��
	bool Step();

	// ɾ��
	bool Delete();

	// ����
	bool IsUpdate();

	// ��ȡ������������ָ��
	void* getInput();

	// ��ȡ�����������ָ��
	void* getOutput();

public:
	// �����㷨���루������ɻ�ʵ�����㷨ģ�鷢�͵ı���״̬��̬�����ݣ�
	AIPilotInput	m_Input;

	// �����㷨��������㷨ģ����ɻ�ʵ�巢�͵ĸ������ָ�����ݣ�
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
