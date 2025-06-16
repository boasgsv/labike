import subprocess
import time

estagios = ['1st', '2nd', '3rd', '4th']
folds = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
tarefas = ['collectionType', 'collectionSite', 'bioActivity', 'collectionSpecie', 'name']

estagio = '1st'

for tarefa in tarefas:
    for fold in folds:
        with open('nohups/' + tarefa + '_' + estagio + '_' + fold +'.out', 'w') as outfile:
            subprocess.run(
                ['nohup', 'python3', 'bike_ft.py', '--tarefa', tarefa, '--estagio', estagio, '--fold', fold],
                stdout=outfile,
                stderr=subprocess.STDOUT
            )
        time.sleep(60)
