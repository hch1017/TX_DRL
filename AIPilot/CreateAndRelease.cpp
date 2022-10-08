#include "CreateAndRelease.h"
#include <stdio.h>
#include <map>

std::map<unsigned short, AIPilot_TeamIntelligame*> g_mapInstances_AIPilot_TeamIntelligame;

AIPilot* CreateInstance(unsigned short index)
{
	// �ڴ˴���Ӵ������Ե��㷨��ʵ������......
	AIPilot_TeamIntelligame* _pInstance = NULL;
	_pInstance = new AIPilot_TeamIntelligame;
	g_mapInstances_AIPilot_TeamIntelligame.insert(std::pair<unsigned short, AIPilot_TeamIntelligame*>(index, _pInstance));

	// ���ش������㷨��ʵ��ָ��
	return (AIPilot*)_pInstance;
}

void ReleaseInstance(unsigned short index)
{
	//if (pInstance != NULL)
	//{
	//	AIPilot_TeamIntelligame* _p = (AIPilot_TeamIntelligame*)pInstance;
	//	delete _p;
	//	_p = NULL;
	//	pInstance = NULL;
	//}
	std::map<unsigned short, AIPilot_TeamIntelligame*>::iterator _iter;
	_iter = g_mapInstances_AIPilot_TeamIntelligame.find(index);
	if (_iter != g_mapInstances_AIPilot_TeamIntelligame.end())
	{
		AIPilot_TeamIntelligame* _pInstance = _iter->second;
		delete _pInstance;
		_pInstance = NULL;
		g_mapInstances_AIPilot_TeamIntelligame.erase(_iter);
	}

	return;
}