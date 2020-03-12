#! /usr/bin/env python
# -*- coding:utf-8 -*-


import rospy
import numpy as np
from geometry_msgs.msg import Twist, Vector3
from sensor_msgs.msg import LaserScan


def scaneou(dado):
	print("Faixa valida: ", dado.range_min , " - ", dado.range_max )
	print("Leituras:")
	print(np.array(dado.ranges).round(decimals=2))
	#print("Intensities")
	#print(np.array(dado.intensities).round(decimals=2))
	recebe_scan = np.array(dado.ranges).round(decimals=2)
	return recebe_scan




if __name__=="__main__":

	rospy.init_node("le_scan")

	velocidade_saida = rospy.Publisher("/cmd_vel", Twist, queue_size = 3 )
	recebe_scan = rospy.Subscriber("/scan", LaserScan, scaneou)

	print(recebe_scan)


	while not rospy.is_shutdown():
		if recebe_scan[0]>=1.02:
			print("TÁ QUASE 1.02")
			stop = Twist(Vector3(0, 0, 0), Vector3(0, 0, 0))
			velocidade_saida.publish(stop)
			rospy.sleep(1)
			velocidade = Twist(Vector3(0.3, 0, 0), Vector3(0, 0, 0))
			velocidade_saida.publish(velocidade)
			rospy.sleep(2)
		elif recebe_scan[0]<1.02 and recebe_scan[0]>0.98:
			print("ENTRE 1.02 E 0.98")
			stop = Twist(Vector3(0, 0, 0), Vector3(0, 0, 0))
			velocidade_saida.publish(stop)
			rospy.sleep(1)
			velocidade = Twist(Vector3(0.3, 0, 0), Vector3(0, 0, 0))
			velocidade_saida.publish(velocidade)
			rospy.sleep(1)

		elif recebe_scan[0]<=0.98:
			print("MENOS Q 0.98")
			stop = Twist(Vector3(0, 0, 0), Vector3(0, 0, 0))
			velocidade_saida.publish(stop)
			rospy.sleep(1)
			velocidade = Twist(Vector3(-0.3, 0, 0), Vector3(0, 0, 0))
			velocidade_saida.publish(velocidade)
			rospy.sleep(1)