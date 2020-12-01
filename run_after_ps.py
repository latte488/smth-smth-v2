import subprocess
import schedule


def process_exist(pid):
    process = subprocess.run(['ps', '-p', str(pid), '-h'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out = process.stdout.decode("utf8")
    return out != ''

def job(finish):
    if not process_exist(9183):
        subprocess.run(['nohup', 'bash', 'plan2.sh', '>', 'plan2.log', '&'])
        finish = True



schedule.every(10).minutes.do(job)

finish = False

while not finish:
    schedule.run_pending()
