#include "AIPilot_TeamIntelligame.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <assert.h>
#include <algorithm>
#define _USE_MATH_DEFINES
#include <cmath>
#include <math.h>
#include "prob.h"
#include "add_feature.h"



using namespace std;

AIPilot_TeamIntelligame::AIPilot_TeamIntelligame(void)
{
}

AIPilot_TeamIntelligame::~AIPilot_TeamIntelligame(void)
{
	// 分配的内存和各类对象请在此处进行释放......
}

/////////////////////////////////////////////////////////////////////////////
// 功能：算法初始化
// 输入：
//			1、pInitData	—	初始化数据块指针，转换为AIPilotInit指针后可获取所有的初始化数据
//			2、index		—	当前算法模块所属飞机的索引号（各方根据自身需要去用）
// 输出：
//			true			—	初始化成功
//			false			—	初始化不成功
/////////////////////////////////////////////////////////////////////////////
bool AIPilot_TeamIntelligame::Init(void* pInitData, unsigned short index)
{	//asd
	// 获取算法的初始化数据
	AIPilotInit* _pInitData = (AIPilotInit*)pInitData;

	// 在此添加所有的算法模块初始化操作......
	session_options.SetIntraOpNumThreads(1);
	session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
	// 初始化成功完成，返回true
	return true;
}

/////////////////////////////////////////////////////////////////////////////
// 功能：决策一步，通过m_Input中提供的仿真飞机实体状态信息进行决策，将决策结果写入m_Output中，该Step操作每10ms仿真时间调用一次
// 输出：无
// 输出：成功完成一次决策返回true，否则返回false
// 备注：若智能算法运算相对较慢，则建议通过多线程方式进行调用，对仿真软件整体推进的影响较小
/////////////////////////////////////////////////////////////////////////////
bool AIPilot_TeamIntelligame::Step()
{
	// 在此添加智能算法模块决策相关内容......
	// python 脚本  推理 cpp调用python
	// input m_Input
	// output m_Output
	// 
	// 
	/////////////////////////////////////////////////////////////////////////////
	Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
	const wchar_t* model_path = L"./AIPilots/Intelligame/model.onnx";
	Ort::Session session(env, model_path, session_options);
	
	size_t num_input_nodes = session.GetInputCount();
	size_t num_output_nodes = session.GetOutputCount();
	std::vector<int64_t> input_node_dims;
	std::vector<int64_t> output_node_dims;
	std::vector<int64_t> input_node_dims_sum;
	std::vector<int64_t> output_node_dims_sum;
	int64_t input_node_dims_sum_all{ 1 };
	int64_t output_node_dims_sum_all{ 1 };
	std::vector<std::vector<int64_t>> input_node_dims_vector;
	std::vector<std::vector<int64_t>> output_node_dims_vector;
	std::vector<const char*> input_node_names;
	std::vector<const char*> output_node_names;
	// 获取所有输入层信息
	for (int i = 0; i < num_input_nodes; i++) {
		// 得到输入节点的名称 char*
		char* input_name = session.GetInputName(i, allocator);
		input_node_names.emplace_back(input_name);

		Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
		auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
		// 得到输入节点的数据类型
		ONNXTensorElementDataType type = tensor_info.GetElementType();

		// 得到输入节点的输入维度 std::vector<int64_t>
		input_node_dims = tensor_info.GetShape();
		input_node_dims_vector.emplace_back(input_node_dims);
		int64_t sums{ 1 };
		// 得到输入节点的输入维度和，后面要使用 int64_t
		for (int j = 0; j < input_node_dims.size(); j++) {
			sums *= input_node_dims[j];
		}
		input_node_dims_sum.emplace_back(sums);
		input_node_dims_sum_all *= sums;
	}
	// 迭代所有输出层信息
	for (int i = 0; i < num_output_nodes; i++) {
		// 得到输出节点的名称 char*
		char* output_name = session.GetOutputName(i, allocator);
		output_node_names.emplace_back(output_name);

		Ort::TypeInfo type_info = session.GetOutputTypeInfo(i);
		auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
		// 得到输出节点的数据类型
		ONNXTensorElementDataType type = tensor_info.GetElementType();

		// 得到输出节点的输入维度 std::vector<int64_t>
		output_node_dims = tensor_info.GetShape();
		output_node_dims_vector.emplace_back(output_node_dims);
		int64_t sums{ 1 };
		// 得到输出节点的输入维度和，后面要使用 int64_t
		for (int j = 0; j < output_node_dims.size(); j++) {
			sums *= output_node_dims[j];
		}
		output_node_dims_sum.emplace_back(sums);
		
		output_node_dims_sum_all *= sums;
	}
	std::vector<std::vector<double>> inputs;
	inputs.push_back({});
	std::vector<std::vector<double>> model_inputs;
	model_inputs.push_back({});
	
	ofstream ofs;
	ofs.open("save_observation/data", ios::app);
	//ofstream ofs2;
	//ofs2.open("save_observation/model_file", ios::app);
	SAircraftBasicInfo_AI_Interface basicinfo = m_Input.m_AircraftBasicInfo;
	SAircraftMoveInfo_AI_Interface moveinfo = m_Input.m_AircraftMoveInfo;
	SRadarInfo_AI_Interface radarinfo = m_Input.m_RadarInfo;
	SHMDInfo_AI_Interface hmdinfo = m_Input.m_HMDInfo;
	SFCInfo_AI_Interface fcinfo = m_Input.m_FCInfo;
	SESMInfo_AI_Interface esminfo = m_Input.m_ESMInfo;
	SDASInfo_AI_Interface dasinfo = m_Input.m_DASInfo;
	SAWACSInfo_AI_Interface awacsinfo = m_Input.m_AWACSInfo;
	SMRAAMDataMsgSet_AI_Interface mraamdinfo = m_Input.m_MRAAMDataMsgSet;
	SAAMDataSet_AI_Interface aamdinfo = m_Input.m_AAMDataSet;
	//basicinfo ______   data[0:4]
	//ofs << basicinfo.m_ID << "," << basicinfo.m_lTimeStamp << "," << basicinfo.m_bAlive << "," << basicinfo.m_fFuel << ",";
	
	//ofs << basicinfo.m_ID << ","; //存储数据的时候吧飞机的id也存进去
	//ofs << basicinfo.m_lTimeStamp << ","; //存储数据的时候吧飞机的id也存进去
	inputs[0].emplace_back(basicinfo.m_ID);
	inputs[0].emplace_back(basicinfo.m_lTimeStamp); //inputs[0][0]，ofs_data[1]
	inputs[0].emplace_back(basicinfo.m_bAlive);//inputs[0][0] category
	inputs[0].emplace_back(basicinfo.m_fFuel / 6000.0);
	//moveinfo ______   data[4:28]
	inputs[0].emplace_back(moveinfo.m_dSelfLon/128.0);//inputs[0][3]，ofs_data[4]
	inputs[0].emplace_back(moveinfo.m_dSelfLat/32.0);
	inputs[0].emplace_back(moveinfo.m_dSelfAlt / 11000.0);
	inputs[0].emplace_back(moveinfo.m_fVN / 700.0);
	inputs[0].emplace_back(moveinfo.m_fVU / 700.0);
	inputs[0].emplace_back(moveinfo.m_fVE / 700.0);

	

	inputs[0].emplace_back(moveinfo.m_fAccN / 100.0);//inputs[0][9]，ofs_data[10]
	inputs[0].emplace_back(moveinfo.m_fAccU / 100.0);
	inputs[0].emplace_back(moveinfo.m_fAccE / 100.0);
	inputs[0].emplace_back(moveinfo.m_fAccBX / 700.0);
	inputs[0].emplace_back(moveinfo.m_fAccBY / 700.0);
	inputs[0].emplace_back(moveinfo.m_fAccBZ / 700.0);
	inputs[0].emplace_back(moveinfo.m_fTASpeed / 700.0);//inputs[0][15]，ofs_data[16]
	inputs[0].emplace_back(moveinfo.m_fMach / 2.2);
	inputs[0].emplace_back(moveinfo.m_fNormalAccel / 10.0);
	inputs[0].emplace_back(moveinfo.m_fYaw / 4.0);
	inputs[0].emplace_back(moveinfo.m_fPitch / 4.0);
	inputs[0].emplace_back(moveinfo.m_fRoll / 4.0);//inputs[0][20]，ofs_data[21]
	inputs[0].emplace_back(moveinfo.m_fAlpha / 4.0);
	inputs[0].emplace_back(moveinfo.m_fBeta / 4.0);
	inputs[0].emplace_back(moveinfo.m_fCrab / 4.0);
	inputs[0].emplace_back(moveinfo.m_fOmegaYaw / 15.0);
	inputs[0].emplace_back(moveinfo.m_fOmegaPitch / 15.0);
	inputs[0].emplace_back(moveinfo.m_fOmegaRoll / 15.0);//inputs[0][26]，ofs_data[27]
	
	//radarinfo  只有在超视距模式下有变化
	inputs[0].emplace_back(radarinfo.m_eRadarWorkMode-1);//inputs[0][27],category
	inputs[0].emplace_back(radarinfo.m_fAziScanRange / 30.0 - 1);//category
	inputs[0].emplace_back(radarinfo.m_fEleScanRange);
	inputs[0].emplace_back(radarinfo.m_fAziScanCent);
	inputs[0].emplace_back(radarinfo.m_fEleScanCent);
	inputs[0].emplace_back(radarinfo.m_uiTgtNum / 8.0);//inputs[0][32]，ofs_data[33]

	//radar_target_info  ______   data[34:122]  这里是否需要考虑最大的8个目标？
	for (int radar_num = 0; radar_num < radarinfo.m_uiTgtNum; radar_num++) {
		inputs[0].emplace_back(radarinfo.m_TgtInfo[radar_num].m_uiTgtLot / 8.0);
		inputs[0].emplace_back(radarinfo.m_TgtInfo[radar_num].m_fTgtDis / 10000.0);
		inputs[0].emplace_back(radarinfo.m_TgtInfo[radar_num].m_fTgtAzi / 4.0);
		inputs[0].emplace_back(radarinfo.m_TgtInfo[radar_num].m_fTgtEle / 4.0);
		inputs[0].emplace_back(radarinfo.m_TgtInfo[radar_num].m_fTgtVN / 700.0);
		inputs[0].emplace_back(radarinfo.m_TgtInfo[radar_num].m_fTgtVU / 700.0);
		inputs[0].emplace_back(radarinfo.m_TgtInfo[radar_num].m_fTgtVE / 700.0);
		inputs[0].emplace_back(radarinfo.m_TgtInfo[radar_num].m_fTgtAccN / 100.0);
		inputs[0].emplace_back(radarinfo.m_TgtInfo[radar_num].m_fTgtAccU / 100.0);
		inputs[0].emplace_back(radarinfo.m_TgtInfo[radar_num].m_fTgtAccE / 100.0);
		inputs[0].emplace_back(radarinfo.m_TgtInfo[radar_num].m_fTgtDisDot / 700.0);
	}
	for (int radar_num_left = radarinfo.m_uiTgtNum; radar_num_left < MAX_TGT_NUM; radar_num_left++) {
		inputs[0].emplace_back(0);
		inputs[0].emplace_back(0);
		inputs[0].emplace_back(0);
		inputs[0].emplace_back(0);
		inputs[0].emplace_back(0);
		inputs[0].emplace_back(0);
		inputs[0].emplace_back(0);
		inputs[0].emplace_back(0);
		inputs[0].emplace_back(0);
		inputs[0].emplace_back(0);
		inputs[0].emplace_back(0);
	}

	//ofs2 << inputs[0].size() << ",";
	inputs[0].emplace_back(hmdinfo.m_uiTgtNum / 8.0); //inputs[0][121]，ofs_data[122]
	for (int hmd_num = 0; hmd_num < hmdinfo.m_uiTgtNum; hmd_num++) {
		inputs[0].emplace_back(hmdinfo.m_TgtInfo[hmd_num].m_uiTgtLot);
		inputs[0].emplace_back(hmdinfo.m_TgtInfo[hmd_num].m_fTgtDis / 10000.0);
		inputs[0].emplace_back(hmdinfo.m_TgtInfo[hmd_num].m_fTgtAziC / 4.0);
		inputs[0].emplace_back(hmdinfo.m_TgtInfo[hmd_num].m_fTgtEleC / 4.0);
		inputs[0].emplace_back(hmdinfo.m_TgtInfo[hmd_num].m_fYaw / 4.0);
		inputs[0].emplace_back(hmdinfo.m_TgtInfo[hmd_num].m_fPitch / 4.0);
		inputs[0].emplace_back(hmdinfo.m_TgtInfo[hmd_num].m_fRoll / 4.0);
		inputs[0].emplace_back(hmdinfo.m_TgtInfo[hmd_num].m_fTgtVN / 700.0);
		inputs[0].emplace_back(hmdinfo.m_TgtInfo[hmd_num].m_fTgtVU / 700.0);
		inputs[0].emplace_back(hmdinfo.m_TgtInfo[hmd_num].m_fTgtVE / 700.0);
		inputs[0].emplace_back(hmdinfo.m_TgtInfo[hmd_num].m_fTgtAccN / 100.0);
		inputs[0].emplace_back(hmdinfo.m_TgtInfo[hmd_num].m_fTgtAccU / 100.0);
		inputs[0].emplace_back(hmdinfo.m_TgtInfo[hmd_num].m_fTgtAccE / 100.0);
		inputs[0].emplace_back(hmdinfo.m_TgtInfo[hmd_num].m_fTgtDisDot / 700.0);
	}
	for (int hmd_num_left = hmdinfo.m_uiTgtNum; hmd_num_left < MAX_TGT_NUM; hmd_num_left++) {
		inputs[0].emplace_back(0);
		inputs[0].emplace_back(0);
		inputs[0].emplace_back(0);
		inputs[0].emplace_back(0);
		inputs[0].emplace_back(0);
		inputs[0].emplace_back(0);
		inputs[0].emplace_back(0);
		inputs[0].emplace_back(0);
		inputs[0].emplace_back(0);
		inputs[0].emplace_back(0);
		inputs[0].emplace_back(0);
		inputs[0].emplace_back(0);
		inputs[0].emplace_back(0);
		inputs[0].emplace_back(0);
	}

	//ofs2 << inputs[0].size() << ",";
	inputs[0].emplace_back(fcinfo.m_eAircraftMainState);//inputs[0][234]，ofs_data[235]
	inputs[0].emplace_back(fcinfo.m_fRmax / 30000.0);
	inputs[0].emplace_back(fcinfo.m_fRnoescape / 30000.0);
	inputs[0].emplace_back(fcinfo.m_fRmin / 6000.0);
	inputs[0].emplace_back(fcinfo.m_bINRNG);
	inputs[0].emplace_back(fcinfo.m_bSHOOT);
	inputs[0].emplace_back(fcinfo.m_bWeaponReady);

	//esminfo _____ data[242:283]
	inputs[0].emplace_back(esminfo.m_uiAlarmTgtNum / 8.0);//inputs[0][241]，ofs_data[242]
	for (int esm_num = 0; esm_num < esminfo.m_uiAlarmTgtNum; esm_num++) {
		inputs[0].emplace_back(esminfo.m_AlarmTgtInfo[esm_num].m_uiTgtLot);
		inputs[0].emplace_back(esminfo.m_AlarmTgtInfo[esm_num].m_eTgtType - 1);
		inputs[0].emplace_back(esminfo.m_AlarmTgtInfo[esm_num].m_eTgtIff - 1);
		inputs[0].emplace_back(esminfo.m_AlarmTgtInfo[esm_num].m_fTgtAzi / 1800.0);
		inputs[0].emplace_back(esminfo.m_AlarmTgtInfo[esm_num].m_fTgtEle / 1800.0);
	}
	for (int esm_num_left = esminfo.m_uiAlarmTgtNum; esm_num_left < MAX_TGT_NUM; esm_num_left++) {
		inputs[0].emplace_back(0);
		inputs[0].emplace_back(0);
		inputs[0].emplace_back(0);
		inputs[0].emplace_back(0);
		inputs[0].emplace_back(0);
	}

	//ofs2 << inputs[0].size() << ",";
	//dasinfo _____ data[283:308]
	inputs[0].emplace_back(dasinfo.m_uiThreatTgtNum / 8.0);//inputs[0][282]，ofs_data[283]
	for (int das_num = 0; das_num < dasinfo.m_uiThreatTgtNum; das_num++) {
		inputs[0].emplace_back(dasinfo.m_ThreatTgtInfo[das_num].m_uiTgtLot);
		inputs[0].emplace_back(dasinfo.m_ThreatTgtInfo[das_num].m_fTgtAzi / 1800.0);
		inputs[0].emplace_back(dasinfo.m_ThreatTgtInfo[das_num].m_fTgtEle / 1800.0);
	}
	for (int das_num_left = dasinfo.m_uiThreatTgtNum; das_num_left < MAX_TGT_NUM; das_num_left++) {
		inputs[0].emplace_back(0);
		inputs[0].emplace_back(0);
		inputs[0].emplace_back(0);
	}

	//awacsinfo _____ data[308:373]
	inputs[0].emplace_back(awacsinfo.m_uiTgtNum / 8.0);//inputs[0][307]，ofs_data[308]
	for (int awacs_num = 0; awacs_num < awacsinfo.m_uiTgtNum; awacs_num++) {
		inputs[0].emplace_back(awacsinfo.m_TgtInfo[awacs_num].m_uiTgtLot);
		inputs[0].emplace_back(awacsinfo.m_TgtInfo[awacs_num].m_eIFF - 1);
		inputs[0].emplace_back(awacsinfo.m_TgtInfo[awacs_num].m_fTgtLon / 128.0);
		inputs[0].emplace_back(awacsinfo.m_TgtInfo[awacs_num].m_fTgtLat / 32.0);
		inputs[0].emplace_back(awacsinfo.m_TgtInfo[awacs_num].m_fTgtAlt / 11000.0);
		inputs[0].emplace_back(awacsinfo.m_TgtInfo[awacs_num].m_fTgtVN / 700.0);
		inputs[0].emplace_back(awacsinfo.m_TgtInfo[awacs_num].m_fTgtVU / 700.0);
		inputs[0].emplace_back(awacsinfo.m_TgtInfo[awacs_num].m_fTgtVE / 700.0);
	}
	for (int awacs_num_left = awacsinfo.m_uiTgtNum; awacs_num_left < MAX_TGT_NUM; awacs_num_left++) {
		inputs[0].emplace_back(0);
		inputs[0].emplace_back(0);
		inputs[0].emplace_back(0);
		inputs[0].emplace_back(0);
		inputs[0].emplace_back(0);
		inputs[0].emplace_back(0);
		inputs[0].emplace_back(0);
		inputs[0].emplace_back(0);
	}
	//ofs2 << inputs[0].size() << ",";
	//mraamdinfo _____ data[373:428]
	inputs[0].emplace_back(mraamdinfo.m_uiMsgNum / 16.0);//inputs[0][372]，ofs_data[373]
	for (int mraamd_num = 0; mraamd_num < mraamdinfo.m_uiMsgNum; mraamd_num++) {
		inputs[0].emplace_back(mraamdinfo.m_MRAAMDataMsg[mraamd_num].m_uiAAMID);
		inputs[0].emplace_back(mraamdinfo.m_MRAAMDataMsg[mraamd_num].m_bSeekerOpen);
		inputs[0].emplace_back(mraamdinfo.m_MRAAMDataMsg[mraamd_num].m_bCapture);
		inputs[0].emplace_back(mraamdinfo.m_MRAAMDataMsg[mraamd_num].m_dLon / 128.0);
		inputs[0].emplace_back(mraamdinfo.m_MRAAMDataMsg[mraamd_num].m_dLat / 32.0);
		inputs[0].emplace_back(mraamdinfo.m_MRAAMDataMsg[mraamd_num].m_dAlt / 10000.0);
		inputs[0].emplace_back(mraamdinfo.m_MRAAMDataMsg[mraamd_num].m_fMslVX / 500.0);
		inputs[0].emplace_back(mraamdinfo.m_MRAAMDataMsg[mraamd_num].m_fMslVU / 1000.0);
		inputs[0].emplace_back(mraamdinfo.m_MRAAMDataMsg[mraamd_num].m_fMslVE / 1000.0);
	}
	for (int mraamd_num_left = mraamdinfo.m_uiMsgNum; mraamd_num_left < MAX_MRAAM_NUM; mraamd_num_left++) {
		inputs[0].emplace_back(0);
		inputs[0].emplace_back(0);
		inputs[0].emplace_back(0);
		inputs[0].emplace_back(0);
		inputs[0].emplace_back(0);
		inputs[0].emplace_back(0);
		inputs[0].emplace_back(0);
		inputs[0].emplace_back(0);
		inputs[0].emplace_back(0);
	}

	//ofs2 << inputs[0].size() << ",";
	//additional feature _____ data[428:430]
	//1.speed
	double speed = sqrt((pow(moveinfo.m_fVN, 2) + pow(moveinfo.m_fVU, 2) + pow(moveinfo.m_fVE, 2)));
	inputs[0].emplace_back(speed / 500.0);
	//2.angle advantage
	double transformedPositionVectorRed[3] = { cos(hmdinfo.m_TgtInfo[0].m_fTgtEleC * M_PI / 180) * cos(hmdinfo.m_TgtInfo[0].m_fTgtAziC * M_PI / 180),
		cos(hmdinfo.m_TgtInfo[0].m_fTgtEleC * M_PI / 180) * sin(hmdinfo.m_TgtInfo[0].m_fTgtAziC * M_PI / 180),
		sin(hmdinfo.m_TgtInfo[0].m_fTgtEleC * M_PI / 180) };
	double* res = reward_angle(moveinfo.m_fYaw, moveinfo.m_fPitch, moveinfo.m_fRoll, transformedPositionVectorRed);
	double phiRed = acos((moveinfo.m_fVE * res[0] + moveinfo.m_fVU * res[1] + moveinfo.m_fVN * res[2]) / (speed + 1e-4));
	inputs[0].emplace_back(1 - abs(phiRed/ M_PI));


	//ofs2 << inputs[0].size() << ",";
	//aamdinfo _____ data[429:610]  only for training
	inputs[0].emplace_back(aamdinfo.m_iAAMNum / 12.0);//ofs_data[429]
	for (int aamd_num = 0; aamd_num < aamdinfo.m_iAAMNum; aamd_num++) {
		inputs[0].emplace_back(aamdinfo.m_AAMData[aamd_num].m_uiAAMID);
		inputs[0].emplace_back(aamdinfo.m_AAMData[aamd_num].m_eAAMType);
		inputs[0].emplace_back(aamdinfo.m_AAMData[aamd_num].m_uiPlaneID);
		inputs[0].emplace_back(aamdinfo.m_AAMData[aamd_num].m_eAAMState);
		inputs[0].emplace_back(aamdinfo.m_AAMData[aamd_num].m_bSeekerOpen);
		inputs[0].emplace_back(aamdinfo.m_AAMData[aamd_num].m_bCapture);
		inputs[0].emplace_back(aamdinfo.m_AAMData[aamd_num].m_dLon);
		inputs[0].emplace_back(aamdinfo.m_AAMData[aamd_num].m_dLat);
		inputs[0].emplace_back(aamdinfo.m_AAMData[aamd_num].m_dAlt / 11000.0);
		inputs[0].emplace_back(aamdinfo.m_AAMData[aamd_num].m_fMslVX / 1000.0);
		inputs[0].emplace_back(aamdinfo.m_AAMData[aamd_num].m_fMslVU / 1000.0);
		inputs[0].emplace_back(aamdinfo.m_AAMData[aamd_num].m_fMslVE / 1000.0);
		inputs[0].emplace_back(aamdinfo.m_AAMData[aamd_num].m_fMslYaw / 4.0);
		inputs[0].emplace_back(aamdinfo.m_AAMData[aamd_num].m_fMslPitch / 4.0);
		inputs[0].emplace_back(aamdinfo.m_AAMData[aamd_num].m_fTgtDis / 10000.0);
	}
	for (int aamd_num_left = aamdinfo.m_iAAMNum; aamd_num_left < MAX_AAM_NUM_IN_SCENE; aamd_num_left++) {
		inputs[0].emplace_back(0);
		inputs[0].emplace_back(0);
		inputs[0].emplace_back(0);
		inputs[0].emplace_back(0);
		inputs[0].emplace_back(0);
		inputs[0].emplace_back(0);
		inputs[0].emplace_back(0);
		inputs[0].emplace_back(0);
		inputs[0].emplace_back(0);
		inputs[0].emplace_back(0);
		inputs[0].emplace_back(0);
		inputs[0].emplace_back(0);
		inputs[0].emplace_back(0);
		inputs[0].emplace_back(0);
		inputs[0].emplace_back(0);
	}

	
	//ofs2 << inputs[0].size() << "," << endl;
	for (int i = 0; i < 611; i++) {
		ofs << inputs[0][i] << ",";
	}

	generate_state(inputs[0], model_inputs[0]);
	/*for (int i = 0; i < model_inputs[0].size(); i++) {
		if (i == model_inputs[0].size() - 1) {
			ofs2 << model_inputs[0][i] << endl;
		}
		else {
			ofs2 << model_inputs[0][i] << ",";
		}
	}*/
	std::vector<std::vector<float>> inputs_f;
	vector<float> input_f(model_inputs[0].begin(), model_inputs[0].end());
	inputs_f.push_back(input_f);
	std::vector<Ort::Value> ort_inputs;
	ort_inputs.reserve(num_input_nodes);
	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

	for (size_t i = 0; i < num_input_nodes; i++) {
		/*ofs2 << "input_size:" << inputs[0].size();
		ofs2 << "input_node_dims_sum:" << input_node_dims_sum[i] << ",";
		ofs2 << "input_node_dims_vector[i].size():" << input_node_dims_vector[i].size() << ",";
		ofs2 << "input_node_dims_vector[i].data():";
		for (int j = 0; j < input_node_dims_vector[i].size(); j++) {
			ofs2 << input_node_dims_vector[i][j] << ",";
		}*/
		
		Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, inputs_f[i].data(),
			input_node_dims_sum[i], input_node_dims_vector[i].data(), input_node_dims_vector[i].size());
		assert(input_tensor.IsTensor());
		
		//size_t DimensionsCount = input_tensor.GetTensorTypeAndShapeInfo().GetDimensionsCount();
		//ofs2 << "DimensionsCount:" << DimensionsCount;
		//std::vector<int64_t> tensor_shape = input_tensor.GetTensorTypeAndShapeInfo().GetShape();
		//ofs2 << "tensor_size:" << tensor_shape.size() << "tensor shape:";
		//for (int j = 0; j < tensor_shape.size(); j++) {
		//	ofs2 << tensor_shape[j] << ",";
		//}
		ort_inputs.emplace_back(std::move(input_tensor));
	}
	
	//Infra
	std::vector<Ort::Value> output_tensors = session.Run(Ort::RunOptions{ nullptr }, input_node_names.data(), 
														ort_inputs.data(), num_input_nodes, output_node_names.data(), 
														num_output_nodes);
	//ofs2 << output_tensors.size() << ",";
	assert(output_tensors.size() == 7 && output_tensors[0].IsTensor() && output_tensors[1].IsTensor());
	assert(output_tensors[2].IsTensor() && output_tensors[3].IsTensor() && output_tensors[4].IsTensor());
	assert(output_tensors[5].IsTensor() && output_tensors[6].IsTensor());

	// 获取输出
	float* output0 = output_tensors[0].GetTensorMutableData<float>();
	float* output1 = output_tensors[1].GetTensorMutableData<float>();
	float* output2 = output_tensors[2].GetTensorMutableData<float>();
	float* output3 = output_tensors[3].GetTensorMutableData<float>();
	float* output4 = output_tensors[4].GetTensorMutableData<float>();
	float* output5 = output_tensors[5].GetTensorMutableData<float>();
	float* output6 = output_tensors[6].GetTensorMutableData<float>();
	//float* output7 = output_tensors[7].GetTensorMutableData<float>();
	//float* output8 = output_tensors[8].GetTensorMutableData<float>();

	std::vector<float> fStickLat_result(5);
	std::vector<float> fStickLon_result(5);
	std::vector<float> fThrottle_result(3);
	std::vector<float> fRudder_result(5);
	std::vector<float> scanLine_result(2);
	std::vector<float> scanRange_result(3);
	std::vector<float> wepon_result(2);
	
	for (int i = 0; i < fStickLat_result.size(); i++) {
		fStickLat_result[i] = output0[i];
	}
	//SoftMax(fStickLat_result);
	for (int i = 0; i < fStickLat_result.size(); i++) {
	}
	for (int i = 0; i < fStickLon_result.size(); i++) {
		fStickLon_result[i] = output1[i];
	}
	//SoftMax(fStickLon_result);
	for (int i = 0; i < fThrottle_result.size(); i++) {
		fThrottle_result[i] = output2[i];
	}
	//SoftMax(fThrottle_result);
	for (int i = 0; i < fRudder_result.size(); i++) {
		fRudder_result[i] = output3[i];
	}
	//SoftMax(fRudder_result);
	for (int i = 0; i < scanLine_result.size(); i++) {
		scanLine_result[i] = output4[i];
	}
	//SoftMax(scanLine_result);
	for (int i = 0; i < scanRange_result.size(); i++) {
		scanRange_result[i] = output5[i];
	}
	//SoftMax(scanRange_result);
	for (int i = 0; i < wepon_result.size(); i++) {
		wepon_result[i] = output6[i];
	}
	//SoftMax(wepon_result);

	int choose_fStickLat;
	int choose_fStickLon;
	int choose_fThrottle;
	int choose_fRudder;
	int choose_scanRange;
	int choose_scanLine;
	int choose_wepon;
	bool sample_choice = true;
	if (sample_choice)
	{
		choose_fStickLat = Sample(fStickLat_result);
		choose_fStickLon = Sample(fStickLon_result);
		choose_fThrottle = Sample(fThrottle_result);
		choose_fRudder = Sample(fRudder_result);
		choose_scanRange = Sample(scanRange_result);
		choose_scanLine = Sample(scanLine_result);
		choose_wepon = Sample(wepon_result);
	}
	else
	{
		choose_fStickLat = std::distance(fStickLat_result.begin(), std::max_element(fStickLat_result.begin(),
			fStickLat_result.end()));
		choose_fStickLon = std::distance(fStickLon_result.begin(), std::max_element(fStickLon_result.begin(),
			fStickLon_result.end()));
		choose_fThrottle = std::distance(fThrottle_result.begin(), std::max_element(fThrottle_result.begin(),
			fThrottle_result.end()));
		choose_fRudder = std::distance(fRudder_result.begin(), std::max_element(fRudder_result.begin(),
			fRudder_result.end()));
		choose_scanRange = std::distance(scanRange_result.begin(), std::max_element(scanRange_result.begin(),
			scanRange_result.end()));
		choose_scanLine = std::distance(scanLine_result.begin(), std::max_element(scanLine_result.begin(),
			scanLine_result.end()));
		choose_wepon = std::distance(wepon_result.begin(), std::max_element(wepon_result.begin(),
			wepon_result.end()));
	}

	//prob  = fStickLat_result[choose_fStickLat]

	// 测试用，填写固定的飞行控制指令  
	m_Output.m_FlyCtrlCmd.m_fStickLat	= m_fStickLat_act[choose_fStickLat]; //[-1,-0.5,0,0.5,1]
	m_Output.m_FlyCtrlCmd.m_fStickLon	= m_fStickLon_act[choose_fStickLon]; //[-1,-0.5,0,0.5,1]
	m_Output.m_FlyCtrlCmd.m_fThrottle	= m_fThrottle_act[choose_fThrottle]; // [0,0,5,1]
	m_Output.m_FlyCtrlCmd.m_fRudder		= m_fRudder_act[choose_fRudder]; // [-1,-0.5,0,0.5,1]

	// 测试用，填写固定的火控系统控制指令
	m_Output.m_FCCtrlCmd.m_eMainTaskMode= Enum_AircraftTaskMode_WVR;

	// 测试用，填写固定的雷达控制指令
	m_Output.m_RadarCtrlCmd.m_eRadarOnOff = Enum_RadarOnOff_ON;
	m_Output.m_RadarCtrlCmd.m_eEleScanLine	= ScanLine_act[choose_scanLine];
	m_Output.m_RadarCtrlCmd.m_eAziScanRange	= ScanRange_act[choose_scanRange];

	// 测试用，填写固定的武器控制指令
	m_Output.m_WeaponCtrlCmd.m_bWeaponLaunch = m_bWeaponLaunch_act[choose_wepon];

	
	/////////////////////////////////////////////////////////////////////////////
	//ofs << prob1 << "," << prob2<< ",";
	// output data[608:616]
	ofs << m_Output.m_FlyCtrlCmd.m_fStickLat << "," << m_Output.m_FlyCtrlCmd.m_fStickLon << "," << m_Output.m_FlyCtrlCmd.m_fThrottle << ",";
	ofs << m_Output.m_FlyCtrlCmd.m_fRudder << "," << m_Output.m_FCCtrlCmd.m_eMainTaskMode << "," << m_Output.m_RadarCtrlCmd.m_eEleScanLine << ",";
	ofs << m_Output.m_RadarCtrlCmd.m_eAziScanRange << "," << m_Output.m_WeaponCtrlCmd.m_bWeaponLaunch << ",";

	ofs << fStickLat_result[choose_fStickLat] << "," << fStickLon_result[choose_fStickLon] << "," << fThrottle_result[choose_fThrottle] << ",";
	ofs << fRudder_result[choose_fRudder] << "," << scanLine_result[choose_scanLine] << "," << scanRange_result[choose_scanRange] << "," ;
	ofs << wepon_result[choose_wepon] << endl;

	//endl = '\n'
	// 成功完成一次决策，返回true
	return true;
}

// 删除
bool AIPilot_TeamIntelligame::Delete()
{
	delete this;
	return true;
}

// 更新
bool AIPilot_TeamIntelligame::IsUpdate()
{
	return true;
}

// 获取决策输入数据指针
void* AIPilot_TeamIntelligame::getInput()
{
	return (void*)&m_Input;
}

// 获取决策输出数据指针
void* AIPilot_TeamIntelligame::getOutput()
{
	return (void*)&m_Output;
}