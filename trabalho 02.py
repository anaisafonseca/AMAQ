entrada = [[1,1],[1,-1],[-1,1],[-1,-1]]
mostrar = [[' 1',' 1'],[' 1','-1'],['-1',' 1'],['-1','-1']]
limiar = 0
x = entrada
wNovo = [None,None]


#etapa de treinamento
#inicialização dos pesos
def pesos(y):
    bAnterior = 0
    wAnterior = [0,0]
    for i in range(len(entrada)):
        wNovo[0] = wAnterior[0] + x[i][0] * y[i]
        wNovo[1] = wAnterior[1] + x[i][1] * y[i]
        bNovo = bAnterior + y[i]
        wAnterior = wNovo
        bAnterior = bNovo
    return bNovo


def geral():
    saida = [1,1,1,1]
    y = saida
    #inicialização dos pesos
    bNovo = pesos(y)
    #teste da rede neural treinada
    for i in range(len(entrada)):
        yLiquido = x[i][0] * wNovo[0] + x[i][1] * wNovo[1] + bNovo
        if yLiquido >= limiar:
            y = ' 1'
        else:
            y = '-1'
        print('|',mostrar[i][0],'|',mostrar[i][1],'|',y,'|')
    print()
    print('w:',wNovo[0],',',wNovo[1])
    print('b:',bNovo)


geral()
#alterar os valores da saída para treinar a máquina e obter os resultados desejados
