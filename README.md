# Momo Script — Edição Roteirista

App para corrigir roteiros de manhwa: caixas de painel (YOLO) + texto gerado por IA.
Os trabalhos chegam prontos pela aba **Jobs** — você pega um, corrige e entrega pelo próprio app.

## Instalação (uma vez só)

Pré-requisitos: [Python 3.11+](https://www.python.org/downloads/) (marque "Add to PATH")
e [Git](https://git-scm.com/download/win) (instalação padrão, só clicar next).

```
git clone https://github.com/victorguima12/momo-script-user.git
cd momo-script-user
INSTALL.bat
```

O primeiro run baixa o modelo de detecção (~116 MB) automaticamente.

## Uso diário

1. Abra com `run.bat`.
2. O app se **atualiza sozinho** ao abrir (git pull automático) — não precisa fazer nada.
3. Aba **Jobs**: configure seu nome (uma vez), escolha um job **verde** (disponível),
   clique **Claim & Download** — as imagens baixam sozinhas e o projeto abre na aba Script.
4. Corrija as caixas e o texto na aba Script (Ctrl+S salva local à vontade).
5. Quando terminar, volte na aba **Jobs** e clique **Deliver**.

Job **vermelho** = outra pessoa já pegou. Cinza = já entregue.

## Problemas?

- Fale com o Victor, ou abra a pasta no Claude Code e descreva o erro.
- Nunca edite os arquivos do app — qualquer mudança local pode travar a atualização
  automática (o app avisa se isso acontecer).
