from os import path
import torch
import torch.utils.tensorboard as tb

def test_logging(train_logger, valid_logger):

    """
    Your code here.
    Finish logging the dummy loss and accuracy
    Log the loss every iteration, the accuracy only after each epoch
    Make sure to set global_step correctly, for epoch=0, iteration=0: global_step=0
    Call the loss 'loss', and accuracy 'accuracy' (no slash or other namespace)
    """

    # This is a strongly simplified training loop
    global_step = 0
    for epoch in range(10):
        torch.manual_seed(epoch)

        dummy_train_accuracy = torch.zeros(0)
        for iteration in range(20):
            dummy_train_loss = 0.9**(epoch+iteration/20.)
            dummy_train_accuracy = torch.cat([dummy_train_accuracy, epoch/10. + torch.randn(10)])
            train_logger.add_scalar('loss', dummy_train_loss, global_step=global_step)
            train_logger.add_scalar('accuracy', torch.mean(dummy_train_accuracy), global_step=global_step)
            global_step += 1

        torch.manual_seed(epoch)
        dummy_validation_accuracy = torch.zeros(0)
        for iteration in range(10):
            dummy_validation_accuracy = torch.cat([dummy_validation_accuracy, epoch / 10. + torch.randn(10)])
        valid_logger.add_scalar('accuracy', dummy_validation_accuracy.mean(), global_step=global_step)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('log_dir')
    args = parser.parse_args()
    train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
    valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'test'), flush_secs=1)
    test_logging(train_logger, valid_logger)
