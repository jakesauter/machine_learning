x=[]
y=[]

with open('gst_3x3.txt') as fp:
	for line in fp:
		str = line[4:9] + ' ' + line[12:17] + ' ' + line[20:25]
		entry = [float(e) for e in str.split()]
		x.append(entry)
		y.append(float(line[line.rindex('.')-1:len(line)-1]))

